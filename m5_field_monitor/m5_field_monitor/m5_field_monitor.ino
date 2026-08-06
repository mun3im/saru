// m5_field_monitor.ino -- field-deployable bird gatekeeper monitor for
// M5Stack Core2. Touchscreen boot selector chooses which detector
// ("gatekeeper") runs on Core 0 -- Goertzel or DrongoNet Micro -- while
// Core 1 continuously encodes captured audio to MP3 and saves it to SD,
// same architecture as the m5_goertzel_bench / m5_drongonet_micro_bench
// dual-core benchmark (M5STACK_DUALCORE_BENCHMARK.md), extended for
// real unattended field use: a full MONITOR_DURATION_MS window instead
// of a 30s lab run, detection events + a periodic heartbeat written to
// an SD text log (not just Serial, which isn't recoverable once nobody
// is watching it in the field), then esp_deep_sleep_start() with no
// wake source -- a single bounded test run per deployment, matching the
// design plan (/home/muneim/.claude/plans/inherited-singing-peach.md).
// Both gatekeepers' code is compiled into this one binary regardless of
// which gets chosen at boot -- only the touchscreen choice decides which
// one actually initializes and runs.
//
// MAG_THRESHOLD / BIRD_SCORE_THRESHOLD below are the same interim,
// idle-noise-informed-but-not-signal-calibrated placeholders carried
// from the benchmark work (see M5STACK_DUALCORE_BENCHMARK.md) -- this is
// the explicit hook point for a future live-touchscreen-adjustment
// upgrade, deliberately left as a plain constant for now (out of scope
// for this pass).

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
#include <SD.h>
#include "esp_bt.h"
#include "esp_sleep.h"
#include "audio_pipeline.h"
#include "mp3_encoder.h"
#include "sd_logger.h"

// ==================== Gatekeeper choice ====================
enum GatekeeperChoice { GATEKEEPER_NONE, GATEKEEPER_GOERTZEL, GATEKEEPER_DRONGONET };
GatekeeperChoice g_choice = GATEKEEPER_NONE;

// ==================== Goertzel parameters (ported from goertzel_m4.ino / m5_goertzel_bench.ino) ====
constexpr float SAMPLE_RATE_HZ = 16000.0f;
constexpr int   BLOCK_SIZE     = 16;
constexpr float TARGET_FREQ_HZ = 3000.0f;
constexpr float MAG_THRESHOLD  = 2000.0f;  // interim, idle-noise-informed -- see M5STACK_DUALCORE_BENCHMARK.md, still needs a real-signal calibration pass

const float OMEGA = 2.0f * PI * TARGET_FREQ_HZ / SAMPLE_RATE_HZ;
const float COEFF = 2.0f * cosf(OMEGA);

// ==================== DrongoNet Micro dims ====================
constexpr int MEL_BINS       = 16;
constexpr int TIME_FRAMES    = 184;
constexpr int INPUT_SIZE     = MEL_BINS * TIME_FRAMES;  // 2944
constexpr int WINDOW_SAMPLES = 48000;  // 3s @ 16kHz
constexpr float BIRD_SCORE_THRESHOLD = 0.8f;  // Portenta Nano's calibrated value -- Micro itself uncalibrated on any hardware, placeholder

// ==================== Monitor control ====================
constexpr unsigned long MONITOR_DURATION_MS = 60000UL;  // SMOKE TEST: 1 minute -- restore to 3600000UL (1hr) for the real field run
constexpr int MP3_BITRATE_KBPS = 32;

volatile bool g_detectorTaskDone = false;
volatile bool g_mp3Done          = false;

// ==================== TFLite Micro (Micro model) -- internal SRAM ====
constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];
tflite::MicroInterpreter *interpreterMicro = nullptr;
TfLiteTensor *inputMicro, *outputMicro;

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
// ARGUS_Common.h (unavailable on this ESP32 core). This is the
// field-deployable sketch, so it gets every applicable win unconditionally
// -- no "regular baseline" concern the way m5_drongonet_mel.ino has.
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

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

// ==================== Touchscreen boot selector ====================
// Plain int constants, not a struct -- Arduino's auto-generated function
// prototypes (inserted at the top of the translation unit, before any
// type defined later in this .ino is visible) broke on a custom struct
// parameter type here; sidestepping that quirk entirely is simpler than
// working around it.
constexpr int BTN_GOERTZEL_X = 20,  BTN_GOERTZEL_Y = 80,  BTN_W = 280, BTN_H = 70;
constexpr int BTN_DRONGONET_X = 20, BTN_DRONGONET_Y = 170;

static bool pointInBox(int px, int py, int bx, int by, int bw, int bh) {
  return px >= bx && px < bx + bw && py >= by && py < by + bh;
}

static void drawButton(int bx, int by, int bw, int bh, const char *label, uint16_t color) {
  M5.Display.fillRect(bx, by, bw, bh, color);
  M5.Display.drawRect(bx, by, bw, bh, TFT_WHITE);
  M5.Display.setTextColor(TFT_WHITE, color);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(bx + 10, by + bh / 2 - 8);
  M5.Display.print(label);
}

GatekeeperChoice runTouchSelector() {
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(20, 20);
  M5.Display.print("Select gatekeeper:");

  drawButton(BTN_GOERTZEL_X, BTN_GOERTZEL_Y, BTN_W, BTN_H, "GOERTZEL", TFT_BLUE);
  drawButton(BTN_DRONGONET_X, BTN_DRONGONET_Y, BTN_W, BTN_H, "DRONGONET MICRO", TFT_BLUE);

  GatekeeperChoice choice = GATEKEEPER_NONE;
  while (choice == GATEKEEPER_NONE) {
    M5.update();
    if (M5.Touch.getCount() > 0) {
      auto detail = M5.Touch.getDetail(0);
      if (detail.wasPressed()) {
        if (pointInBox(detail.x, detail.y, BTN_GOERTZEL_X, BTN_GOERTZEL_Y, BTN_W, BTN_H)) {
          choice = GATEKEEPER_GOERTZEL;
        } else if (pointInBox(detail.x, detail.y, BTN_DRONGONET_X, BTN_DRONGONET_Y, BTN_W, BTN_H)) {
          choice = GATEKEEPER_DRONGONET;
        }
      }
    }
    delay(20);
  }

  const char *chosenLabel = (choice == GATEKEEPER_GOERTZEL) ? "GOERTZEL" : "DRONGONET MICRO";
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_GREEN, TFT_BLACK);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(20, 40);
  M5.Display.print("Selected:");
  M5.Display.setCursor(20, 70);
  M5.Display.print(chosenLabel);
  M5.Display.setCursor(20, 110);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.print("Monitoring for 1 hour,");
  M5.Display.setCursor(20, 135);
  M5.Display.print("then deep sleep.");

  return choice;
}

// ==================== Goertzel detector task (Core 0) ====================
void goertzelTask(void *param) {
  AudioChunk chunk;
  float peakMagnitudeInWindow = 0.0f;
  unsigned long windowStart = millis();
  unsigned long lastHeartbeat = millis();
  bool birdDetected = false;
  char logLine[128];

  for (;;) {
    if (millis() - g_startMillis > g_benchmarkDurationMs) break;

    if (xQueueReceive(qDetector, &chunk, pdMS_TO_TICKS(100)) != pdTRUE) continue;

    for (int block = 0; block < CHUNK_SAMPLES / BLOCK_SIZE; block++) {
      float s1 = 0.0f, s2 = 0.0f;
      const int16_t *blockSamples = &chunk.samples[block * BLOCK_SIZE];
      for (int n = 0; n < BLOCK_SIZE; n++) {
        float x = (float)blockSamples[n];
        float s0 = x + COEFF * s1 - s2;
        s2 = s1;
        s1 = s0;
      }
      float real = s1 - s2 * cosf(OMEGA);
      float imag = s2 * sinf(OMEGA);
      float magnitude = sqrtf(real * real + imag * imag);

      if (magnitude > peakMagnitudeInWindow) peakMagnitudeInWindow = magnitude;

      if (magnitude > MAG_THRESHOLD) {
        if (!birdDetected) {
          birdDetected = true;
          snprintf(logLine, sizeof(logLine), "DETECT,goertzel,%lu,mag=%.1f", millis(), magnitude);
          sdLoggerLogLine(logLine);
          Serial.println(logLine);
        }
      } else {
        birdDetected = false;
      }
    }

    if (millis() - windowStart >= 1000) {
      Serial.print(F("[GOERTZEL] peak(1s)="));
      Serial.println(peakMagnitudeInWindow, 1);
      peakMagnitudeInWindow = 0.0f;
      windowStart = millis();
    }

    if (millis() - lastHeartbeat >= 60000) {
      snprintf(logLine, sizeof(logLine), "HEARTBEAT,goertzel,%lu", millis());
      sdLoggerLogLine(logLine);
      lastHeartbeat = millis();
    }
  }

  g_detectorTaskDone = true;
  vTaskDelete(nullptr);
}

// ==================== DrongoNet Micro detector task (Core 0) ====================
void drongonetDetectorTask(void *param) {
  constexpr int ACCUM_CAPACITY = 48128;
  int16_t *audioAccum = (int16_t *)heap_caps_malloc(ACCUM_CAPACITY * sizeof(int16_t), MALLOC_CAP_SPIRAM);
  float *melOut = (float *)heap_caps_malloc(INPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!audioAccum || !melOut) {
    Serial.println(F("[ERR] PSRAM allocation failed in drongonetDetectorTask."));
    g_detectorTaskDone = true;
    vTaskDelete(nullptr);
    return;
  }
  int accumCount = 0;
  unsigned long lastHeartbeat = millis();
  char logLine[128];

  AudioChunk chunk;

  for (;;) {
    if (millis() - g_startMillis > g_benchmarkDurationMs) break;

    if (xQueueReceive(qDetector, &chunk, pdMS_TO_TICKS(100)) != pdTRUE) continue;

    if (accumCount + CHUNK_SAMPLES > ACCUM_CAPACITY) accumCount = 0;
    memcpy(&audioAccum[accumCount], chunk.samples, CHUNK_SAMPLES * sizeof(int16_t));
    accumCount += CHUNK_SAMPLES;

    if (accumCount >= WINDOW_SAMPLES) {
      computeMelSpectrogram16(audioAccum, WINDOW_SAMPLES, melOut);

      float inScale, outScale; int32_t inZp, outZp;
      getTensorQuantParams(inputMicro, &inScale, &inZp);
      getTensorQuantParams(outputMicro, &outScale, &outZp);
      for (int i = 0; i < INPUT_SIZE; i++) {
        float scaled = melOut[i] / inScale + inZp;
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        inputMicro->data.int8[i] = (int8_t)scaled;
      }

      TfLiteStatus status = interpreterMicro->Invoke();
      if (status == kTfLiteOk) {
        float probBird = (outputMicro->data.int8[1] - outZp) * outScale;
        Serial.print(F("[DRONGONET] P(bird)="));
        Serial.println(probBird, 4);
        if (probBird > BIRD_SCORE_THRESHOLD) {
          snprintf(logLine, sizeof(logLine), "DETECT,drongonet,%lu,p_bird=%.4f", millis(), probBird);
          sdLoggerLogLine(logLine);
          Serial.println(logLine);
        }
      } else {
        Serial.println(F("[DRONGONET] Invoke() failed"));
      }

      int remainder = accumCount - WINDOW_SAMPLES;
      if (remainder > 0) memmove(audioAccum, &audioAccum[WINDOW_SAMPLES], remainder * sizeof(int16_t));
      accumCount = remainder;
    }

    if (millis() - lastHeartbeat >= 60000) {
      snprintf(logLine, sizeof(logLine), "HEARTBEAT,drongonet,%lu", millis());
      sdLoggerLogLine(logLine);
      lastHeartbeat = millis();
    }
  }

  g_detectorTaskDone = true;
  vTaskDelete(nullptr);
}

// ==================== MP3 + SD task (Core 1) -- ported verbatim from the benchmark sketches ====
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
      const uint8_t *mp3Data = mp3EncoderEncode(&stage[consumed], &outLen);
      if (mp3Data && outLen > 0) {
        if (sdBufUsed + outLen > SD_BUF_CAPACITY) {
          sdLoggerWrite(sdBuf, sdBufUsed);
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
        sdLoggerWrite(sdBuf, sdBufUsed);
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
      sdLoggerWrite(sdBuf, sdBufUsed);
      sdBufUsed = 0;
    }
    memcpy(&sdBuf[sdBufUsed], flushData, flushLen);
    sdBufUsed += flushLen;
  }
  if (sdBufUsed > 0) sdLoggerWrite(sdBuf, sdBufUsed);
  sdLoggerFlush();

  g_mp3Done = true;
  vTaskDelete(nullptr);
}

// ==================== Filename helper -- don't overwrite a previous run ====
static void nextAvailableFilename(char *out, size_t outLen, const char *prefix, const char *ext) {
  for (int i = 0; i < 10000; i++) {
    snprintf(out, outLen, "/%s_%04d.%s", prefix, i, ext);
    if (!SD.exists(out)) return;
  }
  // Fell through 10000 files -- extremely unlikely, just reuse the last one.
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

  Serial.println(F("=== m5_field_monitor: touchscreen gatekeeper selector, M5Stack Core2 ==="));

  if (!M5.Mic.isEnabled()) {
    Serial.println(F("[ERR] M5.Mic not enabled."));
    M5.Display.fillScreen(TFT_RED);
    while (1) { delay(1000); }
  }

  if (!sdLoggerBegin()) {
    Serial.println(F("[ERR] SD.begin failed."));
    M5.Display.fillScreen(TFT_RED);
    while (1) { delay(1000); }
  }
  Serial.println(F("[OK]  SD card mounted."));

  g_choice = runTouchSelector();

  char mp3Filename[32], logFilename[32];
  const char *prefix = (g_choice == GATEKEEPER_GOERTZEL) ? "goertzel" : "drongonet";
  nextAvailableFilename(mp3Filename, sizeof(mp3Filename), prefix, "mp3");
  nextAvailableFilename(logFilename, sizeof(logFilename), prefix, "log");

  if (!sdLoggerOpen(mp3Filename) || !sdLoggerOpenLog(logFilename)) {
    Serial.println(F("[ERR] Could not open MP3/log files for writing."));
    M5.Display.fillScreen(TFT_RED);
    while (1) { delay(1000); }
  }
  Serial.print(F("[OK]  Writing to "));
  Serial.print(mp3Filename);
  Serial.print(F(" and "));
  Serial.println(logFilename);

  char banner[96];
  snprintf(banner, sizeof(banner), "START,%s,duration_ms=%lu",
           (g_choice == GATEKEEPER_GOERTZEL) ? "goertzel" : "drongonet", MONITOR_DURATION_MS);
  sdLoggerLogLine(banner);

  if (!mp3EncoderInit((int)SAMPLE_RATE_HZ, MP3_BITRATE_KBPS)) {
    Serial.println(F("[ERR] mp3EncoderInit failed."));
    while (1) { delay(1000); }
  }

  int detectorQueueDepth;
  if (g_choice == GATEKEEPER_DRONGONET) {
    initMelFFT();
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
    Serial.println(F("[OK]  DrongoNet Micro model ready."));
    detectorQueueDepth = 48;  // sized for the ~377-401ms mel+infer burst, see the benchmark's own numbers
  } else {
    Serial.println(F("[OK]  Goertzel ready."));
    detectorQueueDepth = 8;
  }

  audioPipelineInit((int)SAMPLE_RATE_HZ, detectorQueueDepth, /*mp3QueueDepth=*/32);

  Serial.print(F("Monitor duration: "));
  Serial.print(MONITOR_DURATION_MS / 1000);
  Serial.println(F(" s"));
  Serial.println(F("[OK]  Starting tasks.\n"));

  g_startMillis = millis();
  g_benchmarkDurationMs = MONITOR_DURATION_MS;

  // fanoutTask and mp3SdTask are both pinned to core 1 and must be
  // equal priority -- see argus_freertos_priority_starvation memory /
  // M5STACK_DUALCORE_BENCHMARK.md for why (confirmed empirically:
  // unequal priority let the busy-polling fanoutTask starve mp3SdTask
  // almost entirely).
  xTaskCreatePinnedToCore(fanoutTask, "fanout", 4096, nullptr, 2, nullptr, 1);
  xTaskCreatePinnedToCore(mp3SdTask,  "mp3sd",  8192, nullptr, 2, nullptr, 1);

  if (g_choice == GATEKEEPER_DRONGONET) {
    xTaskCreatePinnedToCore(drongonetDetectorTask, "detector", 8192, nullptr, 2, nullptr, 0);
  } else {
    xTaskCreatePinnedToCore(goertzelTask, "detector", 4096, nullptr, 2, nullptr, 0);
  }
}

void loop() {
  if (g_fanoutDone && g_detectorTaskDone && g_mp3Done) {
    sdLoggerClose();

    char endLine[64];
    snprintf(endLine, sizeof(endLine), "END,%lu,entering_deep_sleep", millis());
    sdLoggerLogLine(endLine);
    sdLoggerCloseLog();

    mp3EncoderClose();

    Serial.println(F("[OK]  Monitor window complete. Entering deep sleep (no wake source -- reset to run again)."));
    M5.Display.fillScreen(TFT_BLACK);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.setTextSize(2);
    M5.Display.setCursor(20, 100);
    M5.Display.print("Monitoring complete.");
    M5.Display.setCursor(20, 130);
    M5.Display.print("Entering deep sleep.");
    delay(500);  // let the message land on screen / Serial flush before power-down

    esp_deep_sleep_start();
  }
  delay(200);
}
