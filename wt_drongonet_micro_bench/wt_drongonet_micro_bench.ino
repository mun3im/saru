// wt_drongonet_micro_bench.ino -- DrongoNet-micro mel + inference latency
// on Wio Terminal (Microchip ATSAMD51P19A, Cortex-M4F @ 120MHz), for
// cross-chip comparison against the Portenta H7 (src/DrongoNet/DrongoNet_Micro)
// and M5Stack Core2 (m5_drongonet_micro_bench) numbers.
//
// Synthetic-input timing bench, not the live-mic pipeline: mel computation
// (FFT + fixed filterbank dot-products) and TFLite Micro's INT8 Invoke()
// both take the same number of cycles regardless of *content* for a fixed
// shape/graph, so a deterministic synthetic clip gives the same latency a
// real captured 3s window would. See m5_drongonet_latency.ino's header for
// the same reasoning on the inference side; this sketch extends it to mel.
//
// DSP: CMSIS-DSP's arm_rfft_fast_f32, same real-optimized RFFT the Portenta
// H7 build uses (src/DrongoNet/DrongoNet_Micro/Drongonet.ino) -- NOT the
// vendored Arduino_CMSIS-DSP Arduino-library wrapper (that one declares
// architectures=mbed* only and won't resolve on this classic, non-mbed
// Seeeduino:samd core). Turns out no wrapper library is needed at all:
// the Seeeduino:samd board package bundles the real CMSIS-DSP headers and
// a precompiled `libarm_cortexM4lf_math.a` directly, and its platform.txt
// already passes `-DARM_MATH_CM4 -D__FPU_PRESENT -mfloat-abi=hard
// -mfpu=fpv4-sp-d16` plus links that lib for every sketch on this board --
// `#include <arm_math.h>` alone is enough, confirmed via a minimal
// `arm_rfft_fast_init_f32()` probe sketch. Profiling (see
// WIO_TERMINAL_DRONGONET_MICRO_LATENCY.md) showed FFT was 80% of mel time
// with the previous arduinoFFT (full N-point complex FFT on a real signal,
// half the work wasted on a zero imaginary part) -- this is the fix for
// that, not the filterbank, which was already sparse (only 931 non-zero
// weights stored/applied below, not a dense 16x513 matrix) and only 7.5%
// of the time even before this change.
//
// TFLite Micro: Arduino_TensorFlowLite (the older official Google/TFLite
// Arduino port), not Chirale_TensorFlowLite -- Chirale's system_setup.cpp
// does `using namespace arduino;` unconditionally for any non-mbed board,
// which only resolves on cores built on the newer ArduinoCore-API (mbed,
// RP2040, newer SAMD/megaAVR cores); the classic Seeeduino:samd core here
// doesn't have that namespace and fails to compile. Arduino_TensorFlowLite's
// system_setup.cpp instead uses the classic core's global-namespace
// RingBufferN, which this core does provide.
//
// Model + mel tables (drongonet-micro.h, mel_tables.h) copied verbatim
// from src/DrongoNet/DrongoNet_Micro/ -- architecture-agnostic flatbuffer
// bytes and filterbank constants, identical across every DrongoNet-micro
// variant in this repo (verified via diff).
//
// LCD: Seeed_Arduino_LCD (TFT_eSPI fork), bundled directly inside the
// Seeeduino:samd board package -- not a separately-installed library, and
// already pre-configured (User_Setup.h) for Wio Terminal's onboard 320x240
// ILI9341 display. Used only to show a plain-text status line for whatever
// stage is currently running; not part of any timed region.

#ifdef abs
#undef abs
#endif

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <arm_math.h>
#include <TFT_eSPI.h>

#include "mel_tables.h"
#include "drongonet-micro.h"

// ==================== Model dims ====================
constexpr int MEL_BINS    = 16;
constexpr int TIME_FRAMES = 184;
constexpr int INPUT_SIZE  = MEL_BINS * TIME_FRAMES;  // 2944
constexpr int WINDOW_SAMPLES = 48000;  // 3s @ 16kHz, matches the original block-capture semantics
constexpr int NUM_RUNS = 100;  // bumped from 10 to match the N=100 rigor now used on
                                // Portenta M4/M7 (h7_drongonet_m4_instrumented,
                                // h7_drongonet_m7_instrumented) -- see
                                // GOERTZEL_VS_DRONGONET_LATENCY.md

// ==================== TFLite Micro arena ====================
// 32 KB matches m5_drongonet_micro_bench's Micro arena; DrongoNet-micro
// uses ~24-36 KB in practice (see DrongoNet_Micro/Drongonet.ino comment).
constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input  = nullptr;
TfLiteTensor *output = nullptr;

// ==================== LCD status ====================
TFT_eSPI tft = TFT_eSPI();

void showStatus(const String &line1, const String &line2 = "") {
  tft.fillScreen(TFT_BLACK);
  tft.setCursor(10, 10, 4);
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  tft.println("DrongoNet-micro bench");
  tft.setCursor(10, 60, 4);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.println(line1);
  if (line2.length()) {
    tft.setCursor(10, 100, 2);
    tft.setTextColor(TFT_YELLOW, TFT_BLACK);
    tft.println(line2);
  }
}

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

// ==================== Synthetic audio ====================
// Deterministic, varied, non-silent int16 tone -- not real audio content,
// but shape/scale-correct, matching the "content doesn't affect timing"
// rationale in the file header.
static int16_t audioBuffer[WINDOW_SAMPLES];

void fillSyntheticAudio(int16_t *buf, int n) {
  for (int i = 0; i < n; i++) {
    buf[i] = (int16_t)(8000.0f * sinf((float)i * 0.017f));
  }
}

// ==================== Mel DSP (CMSIS-DSP arm_rfft_fast_f32-based, ported
// from src/DrongoNet/DrongoNet_Micro/Drongonet.ino's computeMelSpectrogram())
static arm_rfft_fast_instance_f32 fftInstance;
static float hannWindow[MEL_N_FFT];
static float fftBuf[MEL_N_FFT];        // in/out, real+imag packed by CMSIS RFFT
static float powerSpectrum[MEL_N_FFT_BINS];

#define FLAT_WEIGHT_TOTAL 931
typedef struct { int startBin; int width; int offset; } MelFilterFlat;
static float melWeightFlat[FLAT_WEIGHT_TOTAL];
static MelFilterFlat melFiltersFlat[MEL_N_MELS];
static float melFeatures[INPUT_SIZE];

// fast_log10f -- IEEE-754 exponent trick (~10 cycles vs ~150 for libm
// log10f), same bit-trick as ARGUS_Common's argus_fast_log10f (copied
// directly rather than #include-ing ARGUS_Common.h, which unconditionally
// pulls in <mbed.h>/<mbed_stats.h> -- not available on this non-mbed core).
// Error < 0.5 dB -- negligible after min-max normalisation.
inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

void initMelFFT() {
  for (int i = 0; i < MEL_N_FFT; i++) {
    hannWindow[i] = 0.5f * (1.0f - cosf(2.0f * (float)PI * i / (float)MEL_N_FFT)) / 32768.0f;
  }

  arm_rfft_fast_init_f32(&fftInstance, MEL_N_FFT);

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

// ==================== Per-phase profiling accumulators ====================
// Reset + read from the sketch each call; micros() overhead (~8 calls/frame
// x 184 frames) is <0.5% of the ~582ms total, negligible for this purpose.
uint32_t g_tWindow = 0, g_tFFT = 0, g_tPower = 0, g_tFilterLog = 0, g_tNorm = 0;

void computeMelSpectrogram16(const int16_t *audio, int audioLength, float *melOut) {
  g_tWindow = g_tFFT = g_tPower = g_tFilterLog = g_tNorm = 0;

  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    uint32_t t0 = micros();
    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) fftBuf[i] = (float)src[i] * hannWindow[i];
    if (validSamples < MEL_N_FFT) memset(&fftBuf[validSamples], 0, (MEL_N_FFT - validSamples) * sizeof(float));
    // No separate imaginary buffer to zero -- arm_rfft_fast_f32 takes a
    // real-only input in-place, unlike arduinoFFT's full-complex transform.
    uint32_t t1 = micros();
    g_tWindow += t1 - t0;

    arm_rfft_fast_f32(&fftInstance, fftBuf, fftBuf, 0 /*forward*/);
    uint32_t t2 = micros();
    g_tFFT += t2 - t1;

    // Power spectrum |X[k]|^2, k = 0..N/2. CMSIS RFFT packs DC into
    // fftBuf[0], Nyquist into fftBuf[1], then complex pairs from fftBuf[2].
    powerSpectrum[0] = fftBuf[0] * fftBuf[0];
    powerSpectrum[MEL_N_FFT / 2] = fftBuf[1] * fftBuf[1];
    arm_cmplx_mag_squared_f32(&fftBuf[2], &powerSpectrum[1], MEL_N_FFT / 2 - 1);
    uint32_t t3 = micros();
    g_tPower += t3 - t2;

    float *melRow = &melOut[f * MEL_N_MELS];
    for (int m = 0; m < MEL_N_MELS; m++) {
      const float *P = &powerSpectrum[melFiltersFlat[m].startBin];
      const float *W = &melWeightFlat[melFiltersFlat[m].offset];
      int width = melFiltersFlat[m].width;
      float acc = 0.0f;
      for (int k = 0; k < width; k++) acc += P[k] * W[k];
      melRow[m] = 10.0f * fastLog10f(acc + 1e-10f);
    }
    uint32_t t4 = micros();
    g_tFilterLog += t4 - t3;
  }

  uint32_t tn0 = micros();
  float globalMin, globalMax;
  uint32_t minIdx, maxIdx;
  arm_min_f32(melOut, INPUT_SIZE, &globalMin, &minIdx);
  arm_max_f32(melOut, INPUT_SIZE, &globalMax, &maxIdx);
  float range = globalMax - globalMin;
  if (range < 1e-6f) range = 1e-6f;
  float invRange = 1.0f / range;
  arm_offset_f32(melOut, -globalMin, melOut, INPUT_SIZE);  // SIMD subtract
  arm_scale_f32(melOut, invRange, melOut, INPUT_SIZE);     // SIMD multiply
  for (int i = 0; i < INPUT_SIZE; i++) {
    float v = melOut[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    melOut[i] = v;
  }
  g_tNorm = micros() - tn0;
}

// ==================== Latency report helper ====================
void reportStats(const char *name, uint32_t *times, uint32_t n) {
  uint32_t sum = 0, mn = UINT32_MAX, mx = 0;
  for (uint32_t i = 0; i < n; i++) {
    sum += times[i];
    if (times[i] < mn) mn = times[i];
    if (times[i] > mx) mx = times[i];
  }
  float avg = (float)sum / (float)n;
  Serial.print(F("["));
  Serial.print(name);
  Serial.print(F("] runs="));
  Serial.print(n);
  Serial.print(F("  avg="));
  Serial.print(avg, 1);
  Serial.print(F(" us  min="));
  Serial.print(mn);
  Serial.print(F(" us  max="));
  Serial.print(mx);
  Serial.println(F(" us"));
}

void setup() {
  Serial.begin(115200);
  uint32_t waitStart = millis();
  while (!Serial && millis() - waitStart < 5000) { }
  delay(500);

  Serial.println(F("=== wt_drongonet_micro_bench: DrongoNet-micro on Wio Terminal (SAMD51) ==="));

  tft.init();
  tft.setRotation(3);
  showStatus("Booting...", "Loading model");

  // ── Load model ────────────────────────────────────────────
  const tflite::Model *model = tflite::GetModel(drongonet_micro);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Model load failed / schema mismatch."));
    while (1) { delay(1000); }
  }

  // Same 6-op set as src/DrongoNet/DrongoNet_Micro/Drongonet.ino:
  // MUL, CONV_2D(x2), MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX.
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddMul();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] AllocateTensors() failed -- bump kTensorArenaSize."));
    while (1) { delay(1000); }
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print(F("[OK]  Arena used: "));
  Serial.print((uint32_t)interpreter->arena_used_bytes());
  Serial.print(F(" / "));
  Serial.println(kTensorArenaSize);

  float inputScale, outputScale;
  int32_t inputZp, outputZp;
  getTensorQuantParams(input,  &inputScale,  &inputZp);
  getTensorQuantParams(output, &outputScale, &outputZp);

  // ── Init mel DSP + synthetic audio ───────────────────────
  showStatus("Model loaded", "Init mel FFT...");
  initMelFFT();
  fillSyntheticAudio(audioBuffer, WINDOW_SAMPLES);
  Serial.println(F("[OK]  Mel filterbank + synthetic audio ready."));
  Serial.println(F("[OK]  Running mel + inference passes now.\n"));

  // ── Timed runs ────────────────────────────────────────────
  uint32_t melTimes[NUM_RUNS];
  uint32_t inferTimes[NUM_RUNS];
  uint32_t bothTimes[NUM_RUNS];

  for (int i = 0; i < NUM_RUNS; i++) {
    // LCD update -- outside the timed mel/infer region below, so it
    // doesn't pollute the measured latency.
    showStatus("Running benchmark", "run " + String(i + 1) + " / " + String(NUM_RUNS));

    uint32_t t0 = micros();
    computeMelSpectrogram16(audioBuffer, WINDOW_SAMPLES, melFeatures);
    uint32_t t1 = micros();
    melTimes[i] = t1 - t0;

    if (i == 0) {
      Serial.println(F("  --- mel phase breakdown (run 0, summed over 184 frames) ---"));
      Serial.print(F("    windowing : ")); Serial.print(g_tWindow);    Serial.println(F(" us"));
      Serial.print(F("    FFT       : ")); Serial.print(g_tFFT);       Serial.println(F(" us"));
      Serial.print(F("    power     : ")); Serial.print(g_tPower);     Serial.println(F(" us"));
      Serial.print(F("    filter+log: ")); Serial.print(g_tFilterLog); Serial.println(F(" us"));
      Serial.print(F("    normalize : ")); Serial.print(g_tNorm);      Serial.println(F(" us"));
    }

    for (int j = 0; j < INPUT_SIZE; j++) {
      float scaled = melFeatures[j] / inputScale + inputZp;
      if (scaled >  127.0f) scaled =  127.0f;
      if (scaled < -128.0f) scaled = -128.0f;
      input->data.int8[j] = (int8_t)scaled;
    }

    uint32_t t2 = micros();
    TfLiteStatus status = interpreter->Invoke();
    uint32_t t3 = micros();
    if (status != kTfLiteOk) {
      Serial.print(F("[ERR] Invoke() failed on run "));
      Serial.println(i);
      inferTimes[i] = 0;
    } else {
      inferTimes[i] = t3 - t2;
    }

    bothTimes[i] = melTimes[i] + inferTimes[i];  // mel + inference (excludes quantize-fill step)

    Serial.print(F("  run "));
    Serial.print(i);
    Serial.print(F(": mel="));
    Serial.print(melTimes[i]);
    Serial.print(F("us  infer="));
    Serial.print(inferTimes[i]);
    Serial.print(F("us  both="));
    Serial.print(bothTimes[i]);
    Serial.println(F("us"));
  }

  Serial.println();
  reportStats("MEL",   melTimes,   NUM_RUNS);
  reportStats("CNN",   inferTimes, NUM_RUNS);
  reportStats("BOTH",  bothTimes,  NUM_RUNS);

  Serial.println(F("\n=== DONE ==="));

  uint32_t melAvg = 0, inferAvg = 0;
  for (int i = 0; i < NUM_RUNS; i++) { melAvg += melTimes[i]; inferAvg += inferTimes[i]; }
  melAvg /= NUM_RUNS;
  inferAvg /= NUM_RUNS;
  showStatus("DONE (" + String(NUM_RUNS) + " runs)",
             "mel=" + String(melAvg) + "us cnn=" + String(inferAvg) + "us");
}

void loop() { delay(1000); }
