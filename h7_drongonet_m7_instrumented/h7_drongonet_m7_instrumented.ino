// h7_drongonet_m7_instrumented.ino — all three DrongoNet variants
// (Nano/Micro/Edge), single-core M7, no RPC, with DWT-cycle-accurate
// latency (and optional INA219 energy) profiling per variant.
//
// Same captured 3s audio window is fed to all three models each loop, so
// their mel-computation cost, inference cost, and (if wired) energy can be
// compared directly against each other on the same hardware/input.
//
// - Nano: FLOAT32 graph I/O boundary (see the dtype note below) --
//   discovered the hard way debugging h7_drongonet_m7. 8-op resolver
//   (explicit QUANTIZE/DEQUANTIZE boundary ops).
// - Micro: genuine INT8 graph I/O (the "folded" production export --
//   src/DrongoNet/DrongoNet_Micro/Drongonet.ino), 6-op resolver, no
//   QUANTIZE/DEQUANTIZE ops needed.
// - Edge: also genuine INT8 I/O, 80-mel input, AllOpsResolver (matching
//   its own production sketch), 308 KB tensor arena in SDRAM -- the 308
//   KB doesn't fit in the Portenta's ~454 KB internal SRAM alongside
//   everything else. Requires a full Portenta H7 (SDRAM-equipped), not
//   the Lite variant.
//
// Nano and Micro share the exact same 16-mel filterbank (verified
// byte-identical against src/DrongoNet/DrongoNet_Micro/mel_tables.h), so
// their mel spectrogram is computed once per loop and reused for both --
// their reported mel latency is therefore identical by construction, not
// a coincidence. Edge uses its own 80-mel filterbank
// (mel_tables_80_edge.h, renamed from src/DrongoNet/DrongoNet_Edge's copy
// to avoid symbol collisions) and gets its own independent mel-timing
// pass.
//
// BIRD_SCORE_THRESHOLD (0.8, empirically tuned against this room's
// aircon-hum baseline) only applies to Nano -- Micro and Edge have no
// calibrated threshold here, so their raw P(bird) scores are reported for
// comparison without a detection verdict. Per mun3im/drongonet's
// MODEL_REFERENCE.md, thresholds do not transfer between variants of
// different capacity/architecture.

#include <arm_math.h>  // CMSIS-DSP, bundled with the Portenta board package

#ifdef abs
#undef abs
#endif

#include "Chirale_TensorFlowLite.h"  // replaces the old Arduino_TensorFlowLite
                                     // (2.4.0-ALPHA, unmaintained since ~2021)
                                     // -- that library's micro_allocator never
                                     // populated tensor->quantization for the
                                     // Nano model (confirmed empirically),
                                     // which silently broke inference.
                                     // Chirale is a current TFLM snapshot,
                                     // Portenta-supported.
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"  // Edge only
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mel_tables.h"           // 16-mel, shared by Nano + Micro
#include "mel_tables_80_edge.h"   // 80-mel, Edge only (renamed symbols)
#include "drongonet-nano.h"
#include "drongonet-micro.h"
#include "seabadnet-edge.h"       // array name inside is drongonet_edge[]

#include <SDRAM.h>  // Edge's tensor arena -- full Portenta H7 only (8 MB @ 0x60000000)

// DWT cycle-counter macros -- same values as
// libraries/ARGUS_Common/src/ARGUS_Common.h (DrongoNet_Nano/Micro, MynaNet,
// ARGUS.ino all use this exact pattern), inlined directly here rather than
// including that header: its unrelated argus_print_system_storage() helper
// declares __etext/__data_start__/__data_end__ with a type/linkage that
// conflicts with this snap's installed mbed_portenta core (3.3.0)'s own
// FlashIAP.h declarations of the same symbols -- a build-environment
// mismatch, not something to route around by editing ARGUS_Common.h itself.
#define DWT_ENABLE()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; \
                           DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while(0)
#define DWT_RESET()   do { DWT->CYCCNT = 0; } while(0)
#define DWT_CYCLES()  (DWT->CYCCNT)
#define DWT_US(cyc)   ((float)(cyc) / 480.0f)   // us at 480 MHz

// Optional power/energy instrumentation -- see src/ina219/README.md and
// Guide_02_INA219_Current_Sensor.md. Not currently wired on this bench;
// the sketch detects its absence at boot and just skips energy reporting.
#include <Adafruit_INA219.h>
#include <Wire.h>

Adafruit_INA219 ina219;
bool ina219_ok = false;

// Sparse mel filterbank: trims each triangular filter's near-zero edge
// weights (below SPARSE_MEL_THRESHOLD * peak, peak=1.0 for these unit
// triangles) instead of storing/accumulating the full startBin..endBin
// range. Measured offline: a 0.1 threshold cuts total stored filterbank
// weights from 931 to 840 (~10%) for this 16-mel table. Only affects the
// filterbank accumulation step in computeMelSpectrogram(); the FFT itself
// is unchanged, so any speedup shows up in the [mel] timing, not [infer].
#define USE_SPARSE_MEL 1
#define SPARSE_MEL_THRESHOLD 0.1f

// ==================== Model Dimensions ====================
// Nano + Micro (shared 16-mel input)
const int MEL_BINS    = 16;
const int TIME_FRAMES = 184;
const int INPUT_SIZE  = MEL_BINS * TIME_FRAMES;  // 2944 elements

// Edge (80-mel input)
const int EDGE_MEL_BINS   = EDGE_MEL_N_MELS;  // 80
const int EDGE_TIME_FRAMES = 184;
const int EDGE_INPUT_SIZE  = EDGE_MEL_BINS * EDGE_TIME_FRAMES;  // 14720

// ==================== Audio Settings ====================
const int micPin            = A0;  // matches this bench's actual mic wiring (h7_goertzel_rpc used A0 too)
const int SAMPLE_RATE       = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // 96 000 B BSS -- shared capture, fed to all 3 variants

// RMS energy gate, ADC units (int16 scale +-32767) -- skip inference on
// silent windows. Same starting point as DrongoNet_Nano; recalibrate for
// your mic/gain.
#define RMS_THRESHOLD 500.0f

// avgScore > this -> bird detected. Raised from the 0.6 default: measured
// aircon-hum baseline in this room reads P(bird) 0.62-0.76, overlapping the
// real bird-sound test's 0.68-0.77 -- 0.8 doesn't fully separate the two
// (nothing will, given that much overlap) but cuts obvious false positives.
// Only calibrated for Nano -- see file header note on Micro/Edge.
#define BIRD_SCORE_THRESHOLD 0.8f

// Edge's per-loop stage stalled indefinitely in testing (no output for 4+
// minutes after Micro reported, board still enumerated/not crashed) --
// likely the 308 KB SDRAM-backed arena being uncached for a 25,890-param
// CNN's memory-bandwidth-heavy conv ops, not yet root-caused. Skipping
// Edge's per-loop run (still initialised in setup() so its arena/tensors
// stay verified) so Nano/Micro keep reporting every cycle instead of
// stalling. Flip to 1 once Edge's slow/hung inference is investigated.
#define RUN_EDGE_INFERENCE 0

// Characterization run: average Nano's and Micro's mel/infer/total
// latency over a fixed number of gated (RMS-passing) windows, then
// report both averages once and go idle -- same convention as
// h7_drongonet_m4_instrumented.ino's CHARACTERIZE_COUNT. Originally
// added Micro-only (see git history); extended to also track Nano so
// the Nano row can be re-run at the same N=100 rigor for the journal
// submission's comparison table. Mel is shared/identical between Nano
// and Micro by construction (same filterbank, computed once per loop),
// so only one mel sum is needed -- each model gets its own infer sum.
#define CHARACTERIZE_COUNT 100
static int      characterizeCount         = 0;
static uint32_t characterizeSumMelUs      = 0;
static uint32_t characterizeSumInferUsNano  = 0;
static uint32_t characterizeSumInferUsMicro = 0;
static bool     characterizeDone          = false;

// ==================== TensorFlow Lite -- Nano ====================
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *modelNano            = nullptr;
tflite::MicroInterpreter *interpreterNano = nullptr;
TfLiteTensor *inputNano  = nullptr;
TfLiteTensor *outputNano = nullptr;

// DrongoNet-nano typically uses ~28-40 KB of arena (see
// src/DrongoNet/DrongoNet_Nano's profiling comments). M7 has no RPC/OpenAMP
// DTCM reservation to fight over (unlike M4 -- see drongonet_m4.ino), so
// there's no tight arena ceiling here; 64 KB gives comfortable headroom.
constexpr int kTensorArenaSizeNano = 64 * 1024;
alignas(16) uint8_t tensorArenaNano[kTensorArenaSizeNano];
}  // namespace

// ==================== TensorFlow Lite -- Micro (folded, genuine INT8 I/O) ====================
namespace {
const tflite::Model *modelMicro            = nullptr;
tflite::MicroInterpreter *interpreterMicro = nullptr;
TfLiteTensor *inputMicro  = nullptr;
TfLiteTensor *outputMicro = nullptr;

// Matches src/DrongoNet/DrongoNet_Micro/Drongonet.ino's own arena size.
constexpr int kTensorArenaSizeMicro = 150 * 1024;
alignas(16) uint8_t tensorArenaMicro[kTensorArenaSizeMicro];
}  // namespace

// ==================== TensorFlow Lite -- Edge (genuine INT8 I/O, SDRAM arena) ====================
namespace {
const tflite::Model *modelEdge            = nullptr;
tflite::MicroInterpreter *interpreterEdge = nullptr;
TfLiteTensor *inputEdge  = nullptr;
TfLiteTensor *outputEdge = nullptr;

// Matches src/DrongoNet/DrongoNet_Edge/DrongoNet_Edge.ino's own arena size
// and SDRAM allocation -- too big for internal SRAM alongside everything
// else running here.
constexpr int kTensorArenaSizeEdge = 308 * 1024;
uint8_t *tensorArenaEdge = nullptr;  // filled in setup() via SDRAM.malloc()
}  // namespace

// ==================== Fast log10f approximation ====================
// Same technique as libraries/ARGUS_Common/src/ARGUS_Common.h's
// argus_fast_log10f() (already used in src/DrongoNet/DrongoNet_Nano) --
// inlined directly here rather than including that whole header, since
// its unrelated argus_print_system_storage() helper conflicts with this
// snap's installed core (see the DWT-macro comment above). IEEE-754
// bit-trick: ~10 cycles vs ~150 for libm log10f(), <0.5 dB error --
// negligible after normalisation, and log10f() is called once per
// (frame, mel bin) -- 2944 times per Nano/Micro inference, 14720 for Edge
// -- so this is the single biggest lever on mel-stage time, more than
// the sparse-filterbank pruning tested earlier.
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

// ==================== CMSIS-DSP FFT instance + buffers ====================
// Shared between the 16-mel (Nano/Micro) and 80-mel (Edge) computations --
// same N_FFT=1024 for both, and they run sequentially within one loop
// iteration, never concurrently, so reusing these is safe.
static arm_rfft_fast_instance_f32 fftInstance;

// DTCM placement is safe here (M7-only, no RPC/OpenAMP conflict) -- see
// src/DrongoNet/DrongoNet_Nano for the same pattern.
__attribute__((section(".dtcmram")))
static float hannWindow[MEL_N_FFT];

__attribute__((section(".dtcmram")))
static float fftBuf[MEL_N_FFT];

__attribute__((section(".dtcmram")))
static float powerSpectrum[MEL_N_FFT_BINS];

// ==================== Flat mel filterbank weight tables ====================
// 16-mel (Nano + Micro)
#define FLAT_WEIGHT_TOTAL 931

typedef struct {
  int startBin;
  int width;
  int offset;
} MelFilterFlat;

static float melWeightFlat[FLAT_WEIGHT_TOTAL];
static MelFilterFlat melFiltersFlat[MEL_N_MELS];

// 80-mel (Edge) -- generous upper bound on total flat-weight count (actual
// sum of all 80 filter widths, computed at init), not hand-counted from the
// filter table to avoid a silent miscount.
#define EDGE_FLAT_WEIGHT_MAX 3000

static float edgeMelWeightFlat[EDGE_FLAT_WEIGHT_MAX];
static MelFilterFlat edgeMelFiltersFlat[EDGE_MEL_N_MELS];

// No cross-window smoothing/state -- each 3s capture window is judged
// independently, so back-to-back calls in separate windows each register
// their own detection instead of needing a rolling average to dip back
// below threshold first.

// ===========================================================
// Quantization param helper -- only Micro and Edge need this (genuine
// INT8 graph I/O); Nano's graph I/O is FLOAT32 (see file header note).
// ===========================================================
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

float   inputScaleMicro = 0, outputScaleMicro = 0;
int32_t inputZpMicro = 0,    outputZpMicro = 0;
float   inputScaleEdge = 0, outputScaleEdge = 0;
int32_t inputZpEdge = 0,    outputZpEdge = 0;

// ===========================================================
// CMSIS-DSP FFT + Hann window + flat filterbank initialisation
// ===========================================================
void initMelFFT() {
  for (int i = 0; i < MEL_N_FFT; i++) {
    // Pre-scaled by 1/32768 here so the per-sample mel loop below doesn't
    // redo this same multiply on every one of the 512 window taps, every
    // one of the 184 frames, every inference (94208 wasted multiplies/run).
    hannWindow[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)MEL_N_FFT)) / 32768.0f;
  }

  arm_rfft_fast_init_f32(&fftInstance, MEL_N_FFT);

  // 16-mel filterbank (Nano + Micro)
  int offset = 0;
  int prunedTotal = 0;
  for (int m = 0; m < MEL_N_MELS; m++) {
    int s = melFilters[m].startBin;
    int p = melFilters[m].peakBin;
    int e = melFilters[m].endBin;
    float rise_inv = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv = (e > p) ? 1.0f / (float)(e - p) : 0.0f;

#if USE_SPARSE_MEL
    // Trim leading/trailing bins below threshold instead of storing the
    // full triangle -- same rise/fall math, just a narrower stored range.
    while (s < p) {
      float w = (float)(s - melFilters[m].startBin) * rise_inv;
      if (w >= SPARSE_MEL_THRESHOLD) break;
      s++;
    }
    while (e > p) {
      // Weight at the current last bin (e-1), relative to the ORIGINAL
      // endBin -- fall_inv is fixed from the untrimmed filter definition.
      float wTrim = (float)(melFilters[m].endBin - e + 1) * fall_inv;
      if (wTrim >= SPARSE_MEL_THRESHOLD) break;
      e--;
    }
#endif

    int width = e - s;
    melFiltersFlat[m].startBin = s;
    melFiltersFlat[m].width    = width;
    melFiltersFlat[m].offset   = offset;
    for (int k = s; k < e; k++) {
      float w;
      if (k < p)       w = (float)(k - melFilters[m].startBin) * rise_inv;
      else if (k == p) w = 1.0f;
      else             w = (float)(melFilters[m].endBin - k) * fall_inv;
      melWeightFlat[offset + (k - s)] = w;
    }
    offset += width;
    prunedTotal += (melFilters[m].endBin - melFilters[m].startBin) - width;
  }
#if USE_SPARSE_MEL
  Serial.print(F("[OK]  Sparse mel: pruned "));
  Serial.print(prunedTotal);
  Serial.print(F(" / "));
  Serial.print(offset + prunedTotal);
  Serial.println(F(" filterbank weights."));
#endif

  // 80-mel filterbank (Edge)
  int edgeOffset = 0;
  for (int m = 0; m < EDGE_MEL_N_MELS; m++) {
    int s = edgeMelFilters[m].startBin;
    int p = edgeMelFilters[m].peakBin;
    int e = edgeMelFilters[m].endBin;
    int width = e - s;
    if (edgeOffset + width > EDGE_FLAT_WEIGHT_MAX) {
      Serial.println(F("[ERR] EDGE_FLAT_WEIGHT_MAX too small -- bump it."));
      while (1) { }
    }
    edgeMelFiltersFlat[m].startBin = s;
    edgeMelFiltersFlat[m].width    = width;
    edgeMelFiltersFlat[m].offset   = edgeOffset;
    float rise_inv = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv = (e > p) ? 1.0f / (float)(e - p) : 0.0f;
    for (int k = s; k < e; k++) {
      float w;
      if (k < p)       w = (float)(k - s) * rise_inv;
      else if (k == p) w = 1.0f;
      else             w = (float)(e - k) * fall_inv;
      edgeMelWeightFlat[edgeOffset + (k - s)] = w;
    }
    edgeOffset += width;
  }
  Serial.print(F("[OK]  Edge flat-weight total used: "));
  Serial.print(edgeOffset);
  Serial.print(F(" / "));
  Serial.println(EDGE_FLAT_WEIGHT_MAX);
}

// ===========================================================
// Mel Spectrogram -- 16-mel (Nano + Micro), FFT-based, matching training
// pipeline
// ===========================================================
void computeMelSpectrogram(int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart   = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) {
      fftBuf[i] = (float)src[i] * hannWindow[i];  // hannWindow pre-scaled by 1/32768 in initMelFFT()
    }
    if (validSamples < MEL_N_FFT) {
      memset(&fftBuf[validSamples], 0, (MEL_N_FFT - validSamples) * sizeof(float));
    }

    arm_rfft_fast_f32(&fftInstance, fftBuf, fftBuf, 0 /*forward*/);

    powerSpectrum[0]           = fftBuf[0] * fftBuf[0];
    powerSpectrum[MEL_N_FFT/2] = fftBuf[1] * fftBuf[1];
    arm_cmplx_mag_squared_f32(&fftBuf[2], &powerSpectrum[1], MEL_N_FFT / 2 - 1);

    float *melRow = &melOut[f * MEL_N_MELS];
    for (int m = 0; m < MEL_N_MELS; m++) {
      const float *P = &powerSpectrum[melFiltersFlat[m].startBin];
      const float *W = &melWeightFlat[melFiltersFlat[m].offset];
      int width = melFiltersFlat[m].width;

      float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
      int k = 0;
      for (; k <= width - 4; k += 4) {
        acc0 += P[k+0] * W[k+0];
        acc1 += P[k+1] * W[k+1];
        acc2 += P[k+2] * W[k+2];
        acc3 += P[k+3] * W[k+3];
      }
      float acc = (acc0 + acc1) + (acc2 + acc3);
      for (; k < width; k++) {
        acc += P[k] * W[k];
      }

      melRow[m] = 10.0f * fastLog10f(acc + 1e-10f);
    }
  }

  float globalMin, globalMax;
  uint32_t minIdx, maxIdx;
  arm_min_f32(melOut, INPUT_SIZE, &globalMin, &minIdx);
  arm_max_f32(melOut, INPUT_SIZE, &globalMax, &maxIdx);

  float range = globalMax - globalMin;
  if (range < 1e-6f) range = 1e-6f;
  float invRange = 1.0f / range;

  arm_offset_f32(melOut, -globalMin, melOut, INPUT_SIZE);
  arm_scale_f32(melOut, invRange, melOut, INPUT_SIZE);
  for (int i = 0; i < INPUT_SIZE; i++) {
    float v = melOut[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    melOut[i] = v;
  }
}

// ===========================================================
// Mel Spectrogram -- 80-mel (Edge only)
// ===========================================================
void computeMelSpectrogramEdge(int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < EDGE_TIME_FRAMES; f++) {
    int frameStart   = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) {
      fftBuf[i] = (float)src[i] * hannWindow[i];  // hannWindow pre-scaled by 1/32768 in initMelFFT()
    }
    if (validSamples < MEL_N_FFT) {
      memset(&fftBuf[validSamples], 0, (MEL_N_FFT - validSamples) * sizeof(float));
    }

    arm_rfft_fast_f32(&fftInstance, fftBuf, fftBuf, 0 /*forward*/);

    powerSpectrum[0]           = fftBuf[0] * fftBuf[0];
    powerSpectrum[MEL_N_FFT/2] = fftBuf[1] * fftBuf[1];
    arm_cmplx_mag_squared_f32(&fftBuf[2], &powerSpectrum[1], MEL_N_FFT / 2 - 1);

    float *melRow = &melOut[f * EDGE_MEL_N_MELS];
    for (int m = 0; m < EDGE_MEL_N_MELS; m++) {
      const float *P = &powerSpectrum[edgeMelFiltersFlat[m].startBin];
      const float *W = &edgeMelWeightFlat[edgeMelFiltersFlat[m].offset];
      int width = edgeMelFiltersFlat[m].width;

      float acc = 0.0f;
      for (int k = 0; k < width; k++) {
        acc += P[k] * W[k];
      }

      melRow[m] = 10.0f * fastLog10f(acc + 1e-10f);
    }
  }

  float globalMin, globalMax;
  uint32_t minIdx, maxIdx;
  arm_min_f32(melOut, EDGE_INPUT_SIZE, &globalMin, &minIdx);
  arm_max_f32(melOut, EDGE_INPUT_SIZE, &globalMax, &maxIdx);

  float range = globalMax - globalMin;
  if (range < 1e-6f) range = 1e-6f;
  float invRange = 1.0f / range;

  arm_offset_f32(melOut, -globalMin, melOut, EDGE_INPUT_SIZE);
  arm_scale_f32(melOut, invRange, melOut, EDGE_INPUT_SIZE);
  for (int i = 0; i < EDGE_INPUT_SIZE; i++) {
    float v = melOut[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    melOut[i] = v;
  }
}

// ===========================================================
// Note on I/O dtype: Nano's declared graph input/output tensors
// (`serving_default_input_1:0` and `StatefulPartitionedCall:0`, per a raw
// FlatBuffer schema dump) are FLOAT32, not INT8 -- confirmed by their
// TensorType()==0 (FLOAT32) with no quantization table attached. The
// INT8 tensors (scale=0.00392157/zp=-128 on the input side, matching
// mun3im/drongonet's MODEL_REFERENCE.md) sit *inside* the graph, between
// explicit QUANTIZE/DEQUANTIZE ops -- this is what a TF converter produces
// when full-integer quantization is used without also setting
// inference_input_type/inference_output_type to int8.
//
// Micro (folded) and Edge do NOT have that float32 wrapper -- their
// declared graph I/O tensors are genuinely INT8 (confirmed: their
// production sketches check inputTensor->bytes == INPUT_SIZE, i.e. 1
// byte/element, and use no QUANTIZE/DEQUANTIZE ops at all), so they use
// real manual quantize/dequantize math against their own runtime-read
// scale/zero_point.
// ===========================================================
// Audio Capture (busy-wait, precise 16 kHz)
// ===========================================================
void captureAudio() {
  unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
  unsigned long lastSample = micros();
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (micros() - lastSample < sampleInterval) {
    }
    lastSample = micros();
    int val = analogRead(micPin);
    float sample = ((float)val - 2048.0f) / 2048.0f;
    audioBuffer[i] = (int16_t)(sample * 32767);
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  delay(1000);

  pinMode(micPin, INPUT);
  analogReadResolution(12);  // this board defaults to 10-bit (idle ~512, per
                             // h7_goertzel_rpc/adc_diag_rpc measurements) --
                             // captureAudio()'s -2048/2048 scaling assumes
                             // true 12-bit (0-4095, idle ~2048)
  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, HIGH);  // Start OFF (Active-LOW)

  DWT_ENABLE();

  Wire.begin();
  if (ina219.begin()) {
    ina219_ok = true;
    Serial.println(F("[OK]  INA219 found -- energy will be reported."));
  } else {
    Serial.println(F("[WARN] INA219 not found -- energy reporting skipped."));
  }

  Serial.println(F("=== h7_drongonet_m7_instrumented: Nano + Micro + Edge, single-core M7 ==="));

  // ─────────────────────── Nano ───────────────────────
  modelNano = tflite::GetModel(drongonet_nano);
  if (!modelNano || modelNano->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Nano model load failed or schema mismatch."));
    while (1) { }
  }

  static tflite::MicroMutableOpResolver<8> resolverNano;
  resolverNano.AddQuantize();
  resolverNano.AddMul();
  resolverNano.AddConv2D();
  resolverNano.AddMaxPool2D();
  resolverNano.AddMean();
  resolverNano.AddFullyConnected();
  resolverNano.AddSoftmax();
  resolverNano.AddDequantize();

  static tflite::MicroInterpreter staticInterpreterNano(
      modelNano, resolverNano, tensorArenaNano, kTensorArenaSizeNano);
  interpreterNano = &staticInterpreterNano;

  if (interpreterNano->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] Nano AllocateTensors() failed -- bump its arena."));
    while (1) { }
  }

  inputNano  = interpreterNano->input(0);
  outputNano = interpreterNano->output(0);

  Serial.print(F("[OK]  Nano  input  type="));
  Serial.print((int)inputNano->type);
  Serial.print(F(" bytes="));
  Serial.print(inputNano->bytes);
  Serial.print(F("  output type="));
  Serial.print((int)outputNano->type);
  Serial.print(F(" bytes="));
  Serial.print(outputNano->bytes);
  Serial.print(F("  arena="));
  Serial.print((uint32_t)interpreterNano->arena_used_bytes());
  Serial.print(F("/"));
  Serial.println(kTensorArenaSizeNano);

  // ─────────────────────── Micro ───────────────────────
  modelMicro = tflite::GetModel(drongonet_micro);
  if (!modelMicro || modelMicro->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Micro model load failed or schema mismatch."));
    while (1) { }
  }

  // Folded graph -- 6 ops, no QUANTIZE/DEQUANTIZE. See
  // src/DrongoNet/DrongoNet_Micro/Drongonet.ino's own decoded op list.
  static tflite::MicroMutableOpResolver<6> resolverMicro;
  resolverMicro.AddMul();
  resolverMicro.AddConv2D();
  resolverMicro.AddMaxPool2D();
  resolverMicro.AddMean();
  resolverMicro.AddFullyConnected();
  resolverMicro.AddSoftmax();

  static tflite::MicroInterpreter staticInterpreterMicro(
      modelMicro, resolverMicro, tensorArenaMicro, kTensorArenaSizeMicro);
  interpreterMicro = &staticInterpreterMicro;

  if (interpreterMicro->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] Micro AllocateTensors() failed -- bump its arena."));
    while (1) { }
  }

  inputMicro  = interpreterMicro->input(0);
  outputMicro = interpreterMicro->output(0);
  getTensorQuantParams(inputMicro,  &inputScaleMicro,  &inputZpMicro);
  getTensorQuantParams(outputMicro, &outputScaleMicro, &outputZpMicro);

  Serial.print(F("[OK]  Micro input  type="));
  Serial.print((int)inputMicro->type);
  Serial.print(F(" bytes="));
  Serial.print(inputMicro->bytes);
  Serial.print(F("  scale="));
  Serial.print(inputScaleMicro, 8);
  Serial.print(F(" zp="));
  Serial.print(inputZpMicro);
  Serial.print(F("  output bytes="));
  Serial.print(outputMicro->bytes);
  Serial.print(F(" scale="));
  Serial.print(outputScaleMicro, 8);
  Serial.print(F(" zp="));
  Serial.print(outputZpMicro);
  Serial.print(F("  arena="));
  Serial.print((uint32_t)interpreterMicro->arena_used_bytes());
  Serial.print(F("/"));
  Serial.println(kTensorArenaSizeMicro);

  // ─────────────────────── Edge ───────────────────────
  SDRAM.begin();
  Serial.println(F("[OK]  SDRAM initialised (8 MB at 0x60000000)."));

  uint8_t *raw = (uint8_t *)SDRAM.malloc(kTensorArenaSizeEdge + 15);
  if (!raw) {
    Serial.println(F("[ERR] SDRAM.malloc() failed for Edge's tensor_arena."));
    while (1) { }
  }
  uintptr_t addr    = (uintptr_t)raw;
  uintptr_t aligned = (addr + 15u) & ~15u;
  tensorArenaEdge = (uint8_t *)aligned;
  Serial.print(F("[OK]  Edge tensor_arena in SDRAM at 0x"));
  Serial.print((uint32_t)tensorArenaEdge, HEX);
  Serial.print(F(", size="));
  Serial.print(kTensorArenaSizeEdge / 1024);
  Serial.println(F(" KB."));

  modelEdge = tflite::GetModel(drongonet_edge);
  if (!modelEdge || modelEdge->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Edge model load failed or schema mismatch."));
    while (1) { }
  }

  // AllOpsResolver -- matches DrongoNet_Edge.ino; its architecture (3
  // BatchNorm'd conv blocks + Dense(32) hidden layer) isn't decoded into a
  // minimal op list anywhere in this repo, so use the same safe default
  // its own production sketch already relies on.
  static tflite::AllOpsResolver resolverEdge;
  static tflite::MicroInterpreter staticInterpreterEdge(
      modelEdge, resolverEdge, tensorArenaEdge, kTensorArenaSizeEdge);
  interpreterEdge = &staticInterpreterEdge;

  if (interpreterEdge->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] Edge AllocateTensors() failed -- bump its arena."));
    while (1) { }
  }

  inputEdge  = interpreterEdge->input(0);
  outputEdge = interpreterEdge->output(0);
  getTensorQuantParams(inputEdge,  &inputScaleEdge,  &inputZpEdge);
  getTensorQuantParams(outputEdge, &outputScaleEdge, &outputZpEdge);

  Serial.print(F("[OK]  Edge  input  type="));
  Serial.print((int)inputEdge->type);
  Serial.print(F(" bytes="));
  Serial.print(inputEdge->bytes);
  Serial.print(F("  scale="));
  Serial.print(inputScaleEdge, 8);
  Serial.print(F(" zp="));
  Serial.print(inputZpEdge);
  Serial.print(F("  output bytes="));
  Serial.print(outputEdge->bytes);
  Serial.print(F(" scale="));
  Serial.print(outputScaleEdge, 8);
  Serial.print(F(" zp="));
  Serial.print(outputZpEdge);
  Serial.print(F("  arena="));
  Serial.print((uint32_t)interpreterEdge->arena_used_bytes());
  Serial.print(F("/"));
  Serial.println(kTensorArenaSizeEdge);

  initMelFFT();
  Serial.println(F("[OK]  CMSIS-DSP RFFT + Hann window initialised."));
  Serial.println(F("[OK]  Ready.\n"));
}

void loop() {
  if (characterizeDone) {
    delay(5000);  // characterization complete -- idle, don't keep capturing/inferring
    return;
  }

  // Guards the characterization accumulation below: only count a window
  // toward the N=100 average if BOTH models' Invoke() succeeded on it --
  // otherwise characterizeCount (incremented in Micro's block) and
  // characterizeSumInferUsNano (incremented in Nano's block) could drift
  // out of sync if either model ever failed independently, silently
  // corrupting one model's average against a mismatched sample count.
  bool nanoOkThisWindow = false;

  captureAudio();

  float rmsAccum = 0.0f;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float s = (float)audioBuffer[i];
    rmsAccum += s * s;
  }
  float rms = sqrtf(rmsAccum / (float)BUFFER_SIZE);

  if (rms < RMS_THRESHOLD) {
    Serial.print(F("[GATE] Silent window -- RMS="));
    Serial.print(rms, 1);
    Serial.println(F(" -- inference skipped."));
    return;
  }

  float power_mW = ina219_ok ? ina219.getPower_mW() : 0.0f;

  Serial.print(F("RMS: "));
  Serial.println(rms, 0);

  // ═══════════════════════ Nano + Micro (shared 16-mel) ═══════════════════════
  static float melFeatures[INPUT_SIZE];

  DWT_RESET();
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, melFeatures);
  uint32_t melCycles16 = DWT_CYCLES();
  float melUs16 = DWT_US(melCycles16);

  // --- Nano: FLOAT32 I/O, no manual quantize/dequantize ---
  memcpy(inputNano->data.f, melFeatures, INPUT_SIZE * sizeof(float));

  DWT_RESET();
  TfLiteStatus statusNano = interpreterNano->Invoke();
  uint32_t inferCyclesNano = DWT_CYCLES();
  float inferUsNano = DWT_US(inferCyclesNano);

  if (statusNano == kTfLiteOk) {
    float probBirdNano = outputNano->data.f[1];
    float totalUsNano = melUs16 + inferUsNano;

    Serial.print(F("[NANO]  P(bird)="));
    Serial.print(probBirdNano, 4);
    Serial.print(F("  mel="));
    Serial.print(melUs16, 1);
    Serial.print(F(" us  infer="));
    Serial.print(inferUsNano, 1);
    Serial.print(F(" us  total="));
    Serial.print(totalUsNano, 1);
    Serial.println(F(" us"));

    if (ina219_ok) {
      float eMel   = power_mW * (melUs16    / 1000.0f);
      float eInfer = power_mW * (inferUsNano / 1000.0f);
      Serial.print(F("        energy: power="));
      Serial.print(power_mW, 1);
      Serial.print(F(" mW  mel="));
      Serial.print(eMel, 1);
      Serial.print(F(" uJ  infer="));
      Serial.print(eInfer, 1);
      Serial.print(F(" uJ  total="));
      Serial.print(eMel + eInfer, 1);
      Serial.println(F(" uJ"));
    }

    // Only Nano has a calibrated threshold (see file header note).
    if (probBirdNano > BIRD_SCORE_THRESHOLD) {
      Serial.println(F("BIRD (nano)"));
      digitalWrite(LEDR, LOW);
      delay(500);
      digitalWrite(LEDR, HIGH);
    }

    nanoOkThisWindow = true;
  } else {
    Serial.println(F("[NANO]  [ERR] Invoke() failed."));
  }

  // --- Micro: genuine INT8 I/O, real quantize/dequantize ---
  for (int i = 0; i < INPUT_SIZE; i++) {
    float scaled = melFeatures[i] / inputScaleMicro + inputZpMicro;
    if (scaled >  127.0f) scaled =  127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    inputMicro->data.int8[i] = (int8_t)scaled;
  }

  DWT_RESET();
  TfLiteStatus statusMicro = interpreterMicro->Invoke();
  uint32_t inferCyclesMicro = DWT_CYCLES();
  float inferUsMicro = DWT_US(inferCyclesMicro);

  if (statusMicro == kTfLiteOk) {
    float probBirdMicro = (outputMicro->data.int8[1] - outputZpMicro) * outputScaleMicro;
    float totalUsMicro = melUs16 + inferUsMicro;

    Serial.print(F("[MICRO] P(bird)="));
    Serial.print(probBirdMicro, 4);
    Serial.print(F("  mel="));
    Serial.print(melUs16, 1);
    Serial.print(F(" us (shared w/ nano)  infer="));
    Serial.print(inferUsMicro, 1);
    Serial.print(F(" us  total="));
    Serial.print(totalUsMicro, 1);
    Serial.println(F(" us"));

    if (ina219_ok) {
      float eMel   = power_mW * (melUs16     / 1000.0f);
      float eInfer = power_mW * (inferUsMicro / 1000.0f);
      Serial.print(F("        energy: power="));
      Serial.print(power_mW, 1);
      Serial.print(F(" mW  mel="));
      Serial.print(eMel, 1);
      Serial.print(F(" uJ  infer="));
      Serial.print(eInfer, 1);
      Serial.print(F(" uJ  total="));
      Serial.print(eMel + eInfer, 1);
      Serial.println(F(" uJ"));
    }

    // Only count this window if Nano ALSO succeeded on it (see
    // nanoOkThisWindow's declaration at the top of loop() for why) --
    // otherwise skip accumulating entirely rather than let the two
    // models' sums drift out of sync with characterizeCount.
    if (!characterizeDone && nanoOkThisWindow) {
      characterizeSumMelUs        += (uint32_t)(melUs16 + 0.5f);
      characterizeSumInferUsNano  += (uint32_t)(inferUsNano + 0.5f);
      characterizeSumInferUsMicro += (uint32_t)(inferUsMicro + 0.5f);
      characterizeCount++;
      if (characterizeCount >= CHARACTERIZE_COUNT) {
        characterizeDone = true;
        float avgMel        = (float)characterizeSumMelUs        / characterizeCount;
        float avgInferNano  = (float)characterizeSumInferUsNano  / characterizeCount;
        float avgInferMicro = (float)characterizeSumInferUsMicro / characterizeCount;
        Serial.println();
        Serial.println(F("=== DrongoNet Nano+Micro M7 characterization complete ==="));
        Serial.print(F("[AVG over ")); Serial.print(characterizeCount); Serial.println(F(" inferences]"));
        Serial.print(F("  avg mel (shared):    ")); Serial.print(avgMel, 1); Serial.println(F(" us"));
        Serial.print(F("  NANO  avg infer: ")); Serial.print(avgInferNano, 1);
        Serial.print(F(" us  avg total: ")); Serial.print(avgMel + avgInferNano, 1); Serial.println(F(" us"));
        Serial.print(F("  MICRO avg infer: ")); Serial.print(avgInferMicro, 1);
        Serial.print(F(" us  avg total: ")); Serial.print(avgMel + avgInferMicro, 1); Serial.println(F(" us"));
      }
    }
  } else {
    Serial.println(F("[MICRO] [ERR] Invoke() failed."));
  }

  // ═══════════════════════ Edge (own 80-mel) ═══════════════════════
#if RUN_EDGE_INFERENCE
  static float melFeaturesEdge[EDGE_INPUT_SIZE];

  DWT_RESET();
  computeMelSpectrogramEdge(audioBuffer, BUFFER_SIZE, melFeaturesEdge);
  uint32_t melCyclesEdge = DWT_CYCLES();
  float melUsEdge = DWT_US(melCyclesEdge);

  for (int i = 0; i < EDGE_INPUT_SIZE; i++) {
    float scaled = melFeaturesEdge[i] / inputScaleEdge + inputZpEdge;
    if (scaled >  127.0f) scaled =  127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    inputEdge->data.int8[i] = (int8_t)scaled;
  }

  DWT_RESET();
  TfLiteStatus statusEdge = interpreterEdge->Invoke();
  uint32_t inferCyclesEdge = DWT_CYCLES();
  float inferUsEdge = DWT_US(inferCyclesEdge);

  if (statusEdge == kTfLiteOk) {
    float probBirdEdge = (outputEdge->data.int8[1] - outputZpEdge) * outputScaleEdge;
    float totalUsEdge = melUsEdge + inferUsEdge;

    Serial.print(F("[EDGE]  P(bird)="));
    Serial.print(probBirdEdge, 4);
    Serial.print(F("  mel="));
    Serial.print(melUsEdge, 1);
    Serial.print(F(" us  infer="));
    Serial.print(inferUsEdge, 1);
    Serial.print(F(" us  total="));
    Serial.print(totalUsEdge, 1);
    Serial.println(F(" us"));

    if (ina219_ok) {
      float eMel   = power_mW * (melUsEdge   / 1000.0f);
      float eInfer = power_mW * (inferUsEdge / 1000.0f);
      Serial.print(F("        energy: power="));
      Serial.print(power_mW, 1);
      Serial.print(F(" mW  mel="));
      Serial.print(eMel, 1);
      Serial.print(F(" uJ  infer="));
      Serial.print(eInfer, 1);
      Serial.print(F(" uJ  total="));
      Serial.print(eMel + eInfer, 1);
      Serial.println(F(" uJ"));
    }
  } else {
    Serial.println(F("[EDGE]  [ERR] Invoke() failed."));
  }
#endif  // RUN_EDGE_INFERENCE

  Serial.println();
}
