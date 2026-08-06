// m5_drongonet_micro_bench.ino -- DrongoNet Micro detector (Core 0)
// running concurrently with MP3 encode + SD save (Core 1), on M5Stack
// Core2. Second of two benchmark sketches, compared against
// m5_goertzel_bench (same Core-1 architecture) per
// /home/muneim/.claude/plans/inherited-singing-peach.md.
//
// Detector task: mel computation ported verbatim from
// m5_drongonet_mel.ino's computeMelSpectrogram16()/initMelFFT() (16-mel,
// n_fft=1024, hop=256, arduinoFFT-based -- CMSIS-DSP isn't available on
// this chip), feeding DrongoNet Micro's TFLite Micro setup ported
// verbatim from m5_drongonet_latency.ino (6-op MicroMutableOpResolver,
// 32KB arena in internal SRAM, INT8 I/O via getTensorQuantParams).
//
// Unlike those two sketches' synthetic-input characterization runs,
// this one processes REAL captured audio, continuously: the detector
// task accumulates chunks from qDetector into a 48000-sample tumbling
// window (not overlapping -- each window's audio is only used once,
// matching the original 3s-block capture semantics those sketches
// measured against), runs mel+infer once full, and immediately starts
// accumulating the next window while emitting results for the one just
// finished.

#ifdef abs
#undef abs
#endif

#include "Chirale_TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <arduinoFFT.h>
#include "mel_tables.h"
#include "drongonet-micro.h"

#include <M5Unified.h>
#include <WiFi.h>
#include "esp_bt.h"
#include "audio_pipeline.h"
#include "mp3_encoder.h"
#include "sd_logger.h"

// ==================== Model dims ====================
constexpr int MEL_BINS    = 16;
constexpr int TIME_FRAMES = 184;
constexpr int INPUT_SIZE  = MEL_BINS * TIME_FRAMES;  // 2944
constexpr int WINDOW_SAMPLES = 48000;  // 3s @ 16kHz, matches the original block-capture semantics

// ==================== TFLite Micro (Micro model) -- internal SRAM, matches m5_drongonet_latency.ino ====
constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];
tflite::MicroInterpreter *interpreterMicro = nullptr;
TfLiteTensor *inputMicro, *outputMicro;
float inputScaleMicro, outputScaleMicro;
int32_t inputZpMicro, outputZpMicro;

void getTensorQuantParams(TfLiteTensor *t, float *scale, int32_t *zero_point) {
  if (t->quantization.type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization *aq = (TfLiteAffineQuantization *)t->quantization.params;
    *scale      = aq->scale->data[0];
    *zero_point = aq->zero_point->data[0];
  } else {
    *scale      = t->params.scale;
    *zero_point = t->params.zero_point;
  }
}

// ==================== Mel DSP (ported verbatim from m5_drongonet_mel.ino) ====
static ArduinoFFT<float> FFT;
static float vReal[MEL_N_FFT];
static float vImag[MEL_N_FFT];
static float hannWindow[MEL_N_FFT];
static float powerSpectrum[MEL_N_FFT_BINS];

#define FLAT_WEIGHT_TOTAL 931
typedef struct { int startBin; int width; int offset; } MelFilterFlat;
static float melWeightFlat[FLAT_WEIGHT_TOTAL];
static MelFilterFlat melFiltersFlat[MEL_N_MELS];

void initMelFFT() {
  for (int i = 0; i < MEL_N_FFT; i++) {
    hannWindow[i] = 0.5f * (1.0f - cosf(2.0f * (float)PI * i / (float)MEL_N_FFT)) / 32768.0f;
  }

  int offset = 0;
  for (int m = 0; m < MEL_N_MELS; m++) {
    int s = melFilters[m].startBin, p = melFilters[m].peakBin, e = melFilters[m].endBin;
    float rise_inv = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv = (e > p) ? 1.0f / (float)(e - p) : 0.0f;
    int width = e - s;
    melFiltersFlat[m] = {s, width, offset};
    for (int k = s; k < e; k++) {
      float w;
      if (k < p)       w = (float)(k - s) * rise_inv;
      else if (k == p) w = 1.0f;
      else             w = (float)(e - k) * fall_inv;
      melWeightFlat[offset + (k - s)] = w;
    }
    offset += width;
  }
}

// fastLog10f -- IEEE-754 exponent bit-trick approximation (~10 cycles vs
// ~150 for libm log10f(), error < 0.5 dB). Same technique as
// ARGUS_Common.h's argus_fast_log10f() / the Portenta instrumented
// sketches' fastLog10f(), copied directly rather than #include-ing
// ARGUS_Common.h (unavailable on this ESP32 core -- pulls in
// <mbed.h>/<mbed_stats.h> unconditionally). This mel code was ported
// verbatim from m5_drongonet_mel.ino's original plain-log10f() pass;
// swapped here per DRONGONET_SPARSE_MEL_BENCHMARK.md's finding -- unlike
// that sketch, this one isn't a deliberate "regular baseline" reference
// point, so no toggle needed.
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

// melOut lives in PSRAM (see the caller in detectorTask) -- this
// function is agnostic to where its output buffer lives.
void computeMelSpectrogram16(const int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) vReal[i] = (float)src[i] * hannWindow[i];
    if (validSamples < MEL_N_FFT) memset(&vReal[validSamples], 0, (MEL_N_FFT - validSamples) * sizeof(float));
    memset(vImag, 0, MEL_N_FFT * sizeof(float));

    FFT.setArrays(vReal, vImag, MEL_N_FFT);
    FFT.compute(FFT_FORWARD);

    for (int k = 0; k < MEL_N_FFT_BINS; k++) {
      powerSpectrum[k] = vReal[k] * vReal[k] + vImag[k] * vImag[k];
    }

    float *melRow = &melOut[f * MEL_N_MELS];
    for (int m = 0; m < MEL_N_MELS; m++) {
      const float *P = &powerSpectrum[melFiltersFlat[m].startBin];
      const float *W = &melWeightFlat[melFiltersFlat[m].offset];
      int width = melFiltersFlat[m].width;
      float acc = 0.0f;
      for (int k = 0; k < width; k++) acc += P[k] * W[k];
      melRow[m] = 10.0f * fastLog10f(acc + 1e-10f);
    }
  }

  float globalMin = melOut[0], globalMax = melOut[0];
  int n = MEL_N_MELS * TIME_FRAMES;
  for (int i = 1; i < n; i++) {
    if (melOut[i] < globalMin) globalMin = melOut[i];
    if (melOut[i] > globalMax) globalMax = melOut[i];
  }
  float range = globalMax - globalMin;
  if (range < 1e-6f) range = 1e-6f;
  float invRange = 1.0f / range;
  for (int i = 0; i < n; i++) {
    float v = (melOut[i] - globalMin) * invRange;
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    melOut[i] = v;
  }
}

// ==================== Benchmark control ====================
constexpr unsigned long BENCHMARK_DURATION_MS = 30000;
volatile bool g_detectorDone = false;
volatile bool g_mp3Done      = false;

// ==================== Detector task stats (Core 0) ====================
volatile uint32_t g_windowCount = 0;
volatile uint64_t g_melUsSum = 0;
volatile uint32_t g_melUsMin = UINT32_MAX;
volatile uint32_t g_melUsMax = 0;
volatile uint64_t g_inferUsSum = 0;
volatile uint32_t g_inferUsMin = UINT32_MAX;
volatile uint32_t g_inferUsMax = 0;
volatile uint32_t g_detectionCount = 0;
constexpr float BIRD_SCORE_THRESHOLD = 0.8f;  // Portenta's calibrated Nano value -- Micro is uncalibrated on any hardware, placeholder same as that caveat elsewhere in this repo

void detectorTask(void *param) {
  // PSRAM -- see m5_drongonet_mel.ino's rationale (this session's own
  // finding: classic ESP32's internal DRAM budget for static globals is
  // tighter than nominal, and these buffers are large enough to matter:
  // audioAccum ~94KB, melOut ~11.5KB). Only Core 0 touches PSRAM in this
  // design (Core 1's mp3SdTask buffers are small, internal-SRAM), so
  // there's no cross-core PSRAM bus contention to worry about.
  constexpr int ACCUM_CAPACITY = 48128;  // next multiple of CHUNK_SAMPLES >= WINDOW_SAMPLES
  int16_t *audioAccum = (int16_t *)heap_caps_malloc(ACCUM_CAPACITY * sizeof(int16_t), MALLOC_CAP_SPIRAM);
  float *melOut = (float *)heap_caps_malloc(INPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!audioAccum || !melOut) {
    Serial.println(F("[ERR] PSRAM allocation failed in detectorTask."));
    g_detectorDone = true;
    vTaskDelete(nullptr);
    return;
  }
  int accumCount = 0;

  AudioChunk chunk;

  for (;;) {
    if (millis() - g_startMillis > g_benchmarkDurationMs) break;

    if (xQueueReceive(qDetector, &chunk, pdMS_TO_TICKS(100)) != pdTRUE) continue;

    if (accumCount + CHUNK_SAMPLES > ACCUM_CAPACITY) {
      // Shouldn't happen (ACCUM_CAPACITY has headroom above WINDOW_SAMPLES),
      // but don't silently overflow if it ever does.
      accumCount = 0;
    }
    memcpy(&audioAccum[accumCount], chunk.samples, CHUNK_SAMPLES * sizeof(int16_t));
    accumCount += CHUNK_SAMPLES;

    if (accumCount >= WINDOW_SAMPLES) {
      uint32_t t0 = micros();
      computeMelSpectrogram16(audioAccum, WINDOW_SAMPLES, melOut);
      uint32_t melUs = micros() - t0;

      float inScale, outScale; int32_t inZp, outZp;
      getTensorQuantParams(inputMicro, &inScale, &inZp);
      getTensorQuantParams(outputMicro, &outScale, &outZp);
      for (int i = 0; i < INPUT_SIZE; i++) {
        float scaled = melOut[i] / inScale + inZp;
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        inputMicro->data.int8[i] = (int8_t)scaled;
      }

      uint32_t t1 = micros();
      TfLiteStatus status = interpreterMicro->Invoke();
      uint32_t inferUs = micros() - t1;

      g_windowCount++;
      g_melUsSum += melUs;
      if (melUs < g_melUsMin) g_melUsMin = melUs;
      if (melUs > g_melUsMax) g_melUsMax = melUs;
      g_inferUsSum += inferUs;
      if (inferUs < g_inferUsMin) g_inferUsMin = inferUs;
      if (inferUs > g_inferUsMax) g_inferUsMax = inferUs;

      if (status == kTfLiteOk) {
        float probBird = (outputMicro->data.int8[1] - outZp) * outScale;
        if (probBird > BIRD_SCORE_THRESHOLD) g_detectionCount++;
        Serial.print(F("[DRONGONET] window="));
        Serial.print(g_windowCount);
        Serial.print(F("  P(bird)="));
        Serial.print(probBird, 4);
        Serial.print(F("  mel="));
        Serial.print(melUs);
        Serial.print(F(" us  infer="));
        Serial.print(inferUs);
        Serial.println(F(" us"));
      } else {
        Serial.println(F("[DRONGONET] Invoke() failed"));
      }

      // Tumbling window: shift any samples beyond WINDOW_SAMPLES (there
      // shouldn't be any at CHUNK_SAMPLES=256/WINDOW_SAMPLES=48000, since
      // 48000 is itself a multiple of 256 -- accumCount lands exactly on
      // 48000, not past it) to the front and keep accumulating.
      int remainder = accumCount - WINDOW_SAMPLES;
      if (remainder > 0) {
        memmove(audioAccum, &audioAccum[WINDOW_SAMPLES], remainder * sizeof(int16_t));
      }
      accumCount = remainder;
    }
  }

  g_detectorDone = true;
  vTaskDelete(nullptr);
}

// ==================== MP3 + SD task (identical to m5_goertzel_bench.ino) ====================
constexpr int MP3_BITRATE_KBPS = 32;

volatile uint32_t g_mp3FrameCount = 0;
volatile uint64_t g_mp3EncodeUsSum = 0;
volatile uint32_t g_mp3EncodeUsMin = UINT32_MAX;
volatile uint32_t g_mp3EncodeUsMax = 0;
volatile uint32_t g_writeCount = 0;
volatile uint64_t g_writeUsSum = 0;
volatile uint32_t g_writeUsMin = UINT32_MAX;
volatile uint32_t g_writeUsMax = 0;

static void writeInstrumented(const uint8_t *data, size_t len) {
  if (len == 0) return;
  uint32_t t0 = micros();
  sdLoggerWrite(data, len);
  uint32_t dt = micros() - t0;
  g_writeCount++;
  g_writeUsSum += dt;
  if (dt < g_writeUsMin) g_writeUsMin = dt;
  if (dt > g_writeUsMax) g_writeUsMax = dt;
}

void mp3SdTask(void *param) {
  constexpr int STAGE_CAPACITY = 1024;
  static int16_t stage[STAGE_CAPACITY];
  int stageCount = 0;

  int samplesPerPass = mp3EncoderSamplesPerPass();

  constexpr size_t SD_BUF_CAPACITY = 8192;
  static uint8_t sdBuf[SD_BUF_CAPACITY];
  size_t sdBufUsed = 0;
  unsigned long lastFlush = millis();

  AudioChunk chunk;

  for (;;) {
    if (millis() - g_startMillis > g_benchmarkDurationMs) break;

    if (xQueueReceive(qMp3, &chunk, pdMS_TO_TICKS(100)) != pdTRUE) continue;

    if (stageCount + CHUNK_SAMPLES > STAGE_CAPACITY) stageCount = 0;
    memcpy(&stage[stageCount], chunk.samples, CHUNK_SAMPLES * sizeof(int16_t));
    stageCount += CHUNK_SAMPLES;

    int consumed = 0;
    while (stageCount - consumed >= samplesPerPass) {
      size_t outLen = 0;
      uint32_t t0 = micros();
      const uint8_t *mp3Data = mp3EncoderEncode(&stage[consumed], &outLen);
      uint32_t dt = micros() - t0;

      g_mp3FrameCount++;
      g_mp3EncodeUsSum += dt;
      if (dt < g_mp3EncodeUsMin) g_mp3EncodeUsMin = dt;
      if (dt > g_mp3EncodeUsMax) g_mp3EncodeUsMax = dt;

      if (mp3Data && outLen > 0) {
        if (sdBufUsed + outLen > SD_BUF_CAPACITY) {
          writeInstrumented(sdBuf, sdBufUsed);
          sdBufUsed = 0;
        }
        memcpy(&sdBuf[sdBufUsed], mp3Data, outLen);
        sdBufUsed += outLen;
      }

      consumed += samplesPerPass;
    }

    if (consumed > 0) {
      memmove(stage, &stage[consumed], (stageCount - consumed) * sizeof(int16_t));
      stageCount -= consumed;
    }

    if (millis() - lastFlush >= 1000) {
      if (sdBufUsed > 0) {
        writeInstrumented(sdBuf, sdBufUsed);
        sdBufUsed = 0;
      }
      sdLoggerFlush();
      lastFlush = millis();
    }
  }

  size_t flushLen = 0;
  const uint8_t *flushData = mp3EncoderFlush(&flushLen);
  if (flushData && flushLen > 0) {
    if (sdBufUsed + flushLen > SD_BUF_CAPACITY) {
      writeInstrumented(sdBuf, sdBufUsed);
      sdBufUsed = 0;
    }
    memcpy(&sdBuf[sdBufUsed], flushData, flushLen);
    sdBufUsed += flushLen;
  }
  if (sdBufUsed > 0) writeInstrumented(sdBuf, sdBufUsed);
  sdLoggerFlush();

  g_mp3Done = true;
  vTaskDelete(nullptr);
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);

  Serial.begin(115200);
  uint32_t waitStart = millis();
  while (!Serial && millis() - waitStart < 3000) { }
  delay(500);

  WiFi.mode(WIFI_OFF);
  btStop();

  Serial.println(F("=== m5_drongonet_micro_bench: DrongoNet Micro (core0) + MP3/SD (core1) on M5Stack Core2 ==="));

  if (!M5.Mic.isEnabled()) {
    Serial.println(F("[ERR] M5.Mic not enabled."));
    while (1) { delay(1000); }
  }

  if (!sdLoggerBegin()) {
    Serial.println(F("[ERR] SD.begin failed."));
    while (1) { delay(1000); }
  }
  Serial.println(F("[OK]  SD card mounted."));

  if (!sdLoggerOpen("/drongonet_bench.mp3")) {
    Serial.println(F("[ERR] Could not open /drongonet_bench.mp3 for writing."));
    while (1) { delay(1000); }
  }
  Serial.println(F("[OK]  Opened /drongonet_bench.mp3"));

  if (!mp3EncoderInit(16000, MP3_BITRATE_KBPS)) {
    Serial.println(F("[ERR] mp3EncoderInit failed."));
    while (1) { delay(1000); }
  }
  Serial.print(F("[OK]  MP3 encoder initialised. samples_per_pass="));
  Serial.println(mp3EncoderSamplesPerPass());

  initMelFFT();
  Serial.println(F("[OK]  Mel filterbank initialised."));

  const tflite::Model *modelMicro = tflite::GetModel(drongonet_micro);
  if (!modelMicro || modelMicro->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Micro model load failed."));
    while (1) { delay(1000); }
  }
  static tflite::MicroMutableOpResolver<6> resolverMicro;
  resolverMicro.AddMul();
  resolverMicro.AddConv2D();
  resolverMicro.AddMaxPool2D();
  resolverMicro.AddMean();
  resolverMicro.AddFullyConnected();
  resolverMicro.AddSoftmax();
  static tflite::MicroInterpreter staticInterpreterMicro(
      modelMicro, resolverMicro, tensorArena, kTensorArenaSize);
  interpreterMicro = &staticInterpreterMicro;
  if (interpreterMicro->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] Micro AllocateTensors() failed."));
    while (1) { delay(1000); }
  }
  inputMicro  = interpreterMicro->input(0);
  outputMicro = interpreterMicro->output(0);
  Serial.print(F("[OK]  Micro model ready. arena="));
  Serial.print((uint32_t)interpreterMicro->arena_used_bytes());
  Serial.print(F("/"));
  Serial.println(kTensorArenaSize);

  // qDetector depth 48 (~768ms @ 16ms/chunk): sized for the ~377ms
  // mel+infer burst measured in m5_drongonet_mel.ino/m5_drongonet_latency.ino
  // (idle-chip numbers, ~2x margin over the ~24-chunk minimum) -- a
  // working assumption per the plan, validate against the real overflow
  // counter below.
  audioPipelineInit(16000, /*detectorQueueDepth=*/48, /*mp3QueueDepth=*/32);

  Serial.print(F("Benchmark duration: "));
  Serial.print(BENCHMARK_DURATION_MS / 1000);
  Serial.println(F(" s"));
  Serial.println(F("[OK]  Starting tasks.\n"));

  g_startMillis = millis();
  g_benchmarkDurationMs = BENCHMARK_DURATION_MS;

  // fanoutTask and mp3SdTask are both pinned to core 1 and must be
  // equal priority -- see m5_goertzel_bench.ino's comment on this same
  // line for why (confirmed empirically there: unequal priority let the
  // busy-polling fanoutTask starve mp3SdTask almost entirely).
  xTaskCreatePinnedToCore(fanoutTask,   "fanout",    4096, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(mp3SdTask,    "mp3sd",     8192, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(detectorTask, "detector", 8192, nullptr, 2, nullptr, 0);
}

void loop() {
  if (g_fanoutDone && g_detectorDone && g_mp3Done) {
    sdLoggerClose();
    mp3EncoderClose();

    Serial.println(F("\n=== SUMMARY ==="));

    Serial.print(F("[DRONGONET] windows="));
    Serial.print(g_windowCount);
    if (g_windowCount > 0) {
      Serial.print(F("  mel avg="));
      Serial.print((float)g_melUsSum / (float)g_windowCount, 1);
      Serial.print(F(" us (min="));
      Serial.print(g_melUsMin);
      Serial.print(F(" max="));
      Serial.print(g_melUsMax);
      Serial.print(F(")  infer avg="));
      Serial.print((float)g_inferUsSum / (float)g_windowCount, 1);
      Serial.print(F(" us (min="));
      Serial.print(g_inferUsMin);
      Serial.print(F(" max="));
      Serial.print(g_inferUsMax);
      Serial.println(F(")"));
    } else {
      Serial.println();
    }
    Serial.print(F("[DRONGONET] detections="));
    Serial.println(g_detectionCount);

    Serial.print(F("[MP3] frames="));
    Serial.print(g_mp3FrameCount);
    if (g_mp3FrameCount > 0) {
      Serial.print(F("  encode avg="));
      Serial.print((float)g_mp3EncodeUsSum / (float)g_mp3FrameCount, 1);
      Serial.print(F(" us  min="));
      Serial.print(g_mp3EncodeUsMin);
      Serial.print(F(" us  max="));
      Serial.print(g_mp3EncodeUsMax);
      Serial.println(F(" us"));
    } else {
      Serial.println();
    }

    Serial.print(F("[SD] writes="));
    Serial.print(g_writeCount);
    if (g_writeCount > 0) {
      Serial.print(F("  avg="));
      Serial.print((float)g_writeUsSum / (float)g_writeCount, 1);
      Serial.print(F(" us  min="));
      Serial.print(g_writeUsMin);
      Serial.print(F(" us  max="));
      Serial.print(g_writeUsMax);
      Serial.println(F(" us"));
    } else {
      Serial.println();
    }
    Serial.print(F("[SD] total MP3 bytes written="));
    Serial.println(sdLoggerBytesWritten());

    Serial.print(F("[QUEUE] qDetector drops="));
    Serial.print(g_detectorDropCount);
    Serial.print(F("  qMp3 drops="));
    Serial.println(g_mp3DropCount);

    Serial.println(F("\n=== DONE ==="));

    while (1) { delay(1000); }
  }
  delay(200);
}
