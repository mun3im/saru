// m5_drongonet_mel.ino -- DrongoNet mel-spectrogram-only latency on
// M5Stack Core2 (ESP32-D0WDQ6-V3, dual-core Xtensa LX6, 240MHz).
// Companion to m5_drongonet_latency.ino (inference-only); this measures
// the OTHER half of the pipeline that sketch deliberately skipped.
// Separate sketch per task, per instruction -- mel here, capture in a
// later sketch, not combined into one.
//
// Same 16-mel (Nano/Micro) and 80-mel (Edge) filterbanks as the Portenta
// pipeline (mel_tables.h / mel_tables_80_edge.h, copied verbatim -- same
// n_fft=1024, hop=256, sr=16000 for both, only filterbank width differs).
// Same windowing/log/normalize algorithm as
// h7_drongonet_m7_instrumented.ino's computeMelSpectrogram()/
// computeMelSpectrogramEdge(), ported from CMSIS-DSP (ARM-only, not
// available here) to arduinoFFT -- already vendored in this repo
// (libraries/arduinoFFT, architectures=*, used by the top-level
// mic_fft/mic_magma scratch sketches) rather than a new dependency.
// USE_FASTLOG (below, default 1) selects between the fastlog10f()
// bit-trick optimization already applied on Portenta and the plain
// log10f() this sketch originally shipped with. That original plain-
// log10f() pass is the "regular" baseline M5STACK_INFERENCE_LATENCY.md
// cites (294.0ms for MEL16) -- set USE_FASTLOG 0 to reproduce that exact
// historical figure; the default (1) is the current fastest-known
// mel config per DRONGONET_SPARSE_MEL_BENCHMARK.md's finding, which this
// sketch predates. Note: arduinoFFT computes a full N-point complex FFT
// (real input, zero imaginary), not a real-optimized half-size algorithm
// like CMSIS's arm_rfft_fast_f32 -- expect this to cost more than a
// real-FFT-based implementation would, on top of whatever the chip-level
// gap already is (fastlog doesn't change that FFT-algorithm gap, only
// the log step).
//
// Synthetic audio input (see m5_drongonet_latency.ino's header for why
// content doesn't need to be real captured audio for a latency number --
// same reasoning applies to the DSP stage: FFT/filterbank arithmetic
// cost is fixed by buffer size, not sample values).

#include <arduinoFFT.h>
#include "mel_tables.h"
#include "mel_tables_80_edge.h"

#include "WiFi.h"
#include "esp_bt.h"

// See file header -- 0 reproduces the original plain-log10f() "regular"
// baseline this sketch shipped with (cited as 294.0ms MEL16 in
// M5STACK_INFERENCE_LATENCY.md); 1 (default) uses the fastlog10f()
// bit-trick, the current fastest-known mel config elsewhere in this repo.
#define USE_FASTLOG 1

#if USE_FASTLOG
// fastLog10f -- IEEE-754 exponent bit-trick approximation (~10 cycles vs
// ~150 for libm log10f(), error < 0.5 dB). Same technique as
// ARGUS_Common.h's argus_fast_log10f() / the Portenta instrumented
// sketches' fastLog10f(), copied directly rather than #include-ing
// ARGUS_Common.h (that header unconditionally pulls in
// <mbed.h>/<mbed_stats.h>, unavailable on this ESP32 core).
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}
#define LOG10F(x) fastLog10f(x)
#else
#define LOG10F(x) log10f(x)
#endif

constexpr int SAMPLE_RATE       = 16000;
constexpr int AUDIO_DURATION_MS = 3000;
constexpr int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

constexpr int NUM_RUNS = 5;  // mel-only passes are much cheaper than inference; 5 is plenty

// PSRAM, not internal DRAM -- classic ESP32's internal DRAM budget for
// static globals turned out much tighter than the nominal ~320 KB
// (WiFi/BT/FreeRTOS reserve static BSS space regardless of runtime
// on/off state -- WiFi.mode(WIFI_OFF)/btStop() free heap, not the
// linker-level static footprint). audioBuffer (96 KB) and both mel
// output buffers are read/written throughout the timed region, so
// this isn't cost-free -- PSRAM access latency is now part of what's
// being measured here, not purely internal-SRAM DSP cost. Documented
// as a caveat in the results doc rather than hidden: on a RAM-this-
// constrained chip, a real deployment doing 3s-clip mel computation
// might genuinely need to make the same tradeoff, so it's not a purely
// artificial handicap either.
int16_t *audioBuffer;

static ArduinoFFT<float> FFT;
static float vReal[MEL_N_FFT];
static float vImag[MEL_N_FFT];
static float hannWindow[MEL_N_FFT];  // pre-scaled by 1/32768, same reasoning as the Portenta sketches
static float powerSpectrum[MEL_N_FFT_BINS];

// ==================== 16-mel (Nano/Micro) flat filterbank ====================
#define FLAT_WEIGHT_TOTAL 931
typedef struct { int startBin; int width; int offset; } MelFilterFlat;
static float melWeightFlat[FLAT_WEIGHT_TOTAL];
static MelFilterFlat melFiltersFlat[MEL_N_MELS];
static float *melOut16;  // PSRAM, see audioBuffer's comment above

// ==================== 80-mel (Edge) flat filterbank ====================
#define EDGE_FLAT_WEIGHT_MAX 3000
static float edgeMelWeightFlat[EDGE_FLAT_WEIGHT_MAX];
static MelFilterFlat edgeMelFiltersFlat[EDGE_MEL_N_MELS];
static int edgeFlatWeightUsed = 0;
const int EDGE_TIME_FRAMES = 184;
static float *melOutEdge;  // PSRAM, see audioBuffer's comment above

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

  offset = 0;
  for (int m = 0; m < EDGE_MEL_N_MELS; m++) {
    int s = edgeMelFilters[m].startBin, p = edgeMelFilters[m].peakBin, e = edgeMelFilters[m].endBin;
    float rise_inv = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv = (e > p) ? 1.0f / (float)(e - p) : 0.0f;
    int width = e - s;
    edgeMelFiltersFlat[m] = {s, width, offset};
    for (int k = s; k < e; k++) {
      float w;
      if (k < p)       w = (float)(k - s) * rise_inv;
      else if (k == p) w = 1.0f;
      else             w = (float)(e - k) * fall_inv;
      edgeMelWeightFlat[offset + (k - s)] = w;
    }
    offset += width;
  }
  edgeFlatWeightUsed = offset;
}

void computeMelSpectrogram16(const int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < 184; f++) {
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
      melRow[m] = 10.0f * LOG10F(acc + 1e-10f);
    }
  }

  float globalMin = melOut[0], globalMax = melOut[0];
  int n = MEL_N_MELS * 184;
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

void computeMelSpectrogramEdge(const int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < EDGE_TIME_FRAMES; f++) {
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

    float *melRow = &melOut[f * EDGE_MEL_N_MELS];
    for (int m = 0; m < EDGE_MEL_N_MELS; m++) {
      const float *P = &powerSpectrum[edgeMelFiltersFlat[m].startBin];
      const float *W = &edgeMelWeightFlat[edgeMelFiltersFlat[m].offset];
      int width = edgeMelFiltersFlat[m].width;
      float acc = 0.0f;
      for (int k = 0; k < width; k++) acc += P[k] * W[k];
      melRow[m] = 10.0f * LOG10F(acc + 1e-10f);
    }
  }

  float globalMin = melOut[0], globalMax = melOut[0];
  int n = EDGE_MEL_N_MELS * EDGE_TIME_FRAMES;
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

void fillSyntheticAudio() {
  for (int i = 0; i < BUFFER_SIZE; i++) {
    audioBuffer[i] = (int16_t)(3000.0f * sinf((float)i * 0.083f) + 800.0f * sinf((float)i * 0.31f));
  }
}

void runReport(const char *name, void (*fn)(const int16_t *, int, float *), float *out) {
  uint32_t times[NUM_RUNS];
  uint32_t sum = 0, mn = UINT32_MAX, mx = 0;
  for (int i = 0; i < NUM_RUNS; i++) {
    uint32_t t0 = micros();
    fn(audioBuffer, BUFFER_SIZE, out);
    uint32_t t1 = micros();
    uint32_t dt = t1 - t0;
    times[i] = dt;
    sum += dt;
    if (dt < mn) mn = dt;
    if (dt > mx) mx = dt;
  }
  float avg = (float)sum / (float)NUM_RUNS;
  Serial.print(F("["));
  Serial.print(name);
  Serial.print(F("] avg="));
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
  while (!Serial && millis() - waitStart < 3000) { }
  delay(500);

  WiFi.mode(WIFI_OFF);
  btStop();

  audioBuffer = (int16_t *)heap_caps_malloc(BUFFER_SIZE * sizeof(int16_t), MALLOC_CAP_SPIRAM);
  melOut16    = (float *)heap_caps_malloc(MEL_N_MELS * 184 * sizeof(float), MALLOC_CAP_SPIRAM);
  melOutEdge  = (float *)heap_caps_malloc(80 * 184 * sizeof(float), MALLOC_CAP_SPIRAM);
  if (!audioBuffer || !melOut16 || !melOutEdge) {
    Serial.println(F("[ERR] PSRAM allocation failed."));
    while (1) { delay(1000); }
  }

  Serial.println(F("=== m5_drongonet_mel: 16-mel (Nano/Micro) + 80-mel (Edge) on M5Stack Core2 ==="));
  Serial.print(F("MEL_N_FFT=")); Serial.print(MEL_N_FFT);
  Serial.print(F(" MEL_HOP_LENGTH=")); Serial.print(MEL_HOP_LENGTH);
  Serial.print(F(" MEL_N_MELS=")); Serial.println(MEL_N_MELS);

  initMelFFT();
  Serial.print(F("[OK]  16-mel flat weights: "));
  Serial.println(melFiltersFlat[MEL_N_MELS - 1].offset + melFiltersFlat[MEL_N_MELS - 1].width);
  Serial.print(F("[OK]  80-mel flat weights: "));
  Serial.println(edgeFlatWeightUsed);

  fillSyntheticAudio();

  Serial.println(F("[OK]  Ready -- running mel passes now.\n"));

  runReport("MEL16", computeMelSpectrogram16, melOut16);
  runReport("MEL80_EDGE", computeMelSpectrogramEdge, melOutEdge);

  Serial.println(F("\n=== DONE ==="));
}

void loop() { delay(1000); }
