// drongonet_m4_instrumented.ino — DrongoNet Nano + Micro on M4, RPC to M7
// M4-core counterpart of h7_drongonet_m7_instrumented: same DWT-timed
// mel-only / inference-only / combined latency measurement, for Nano and
// Micro (Edge skipped -- its 308 KB SDRAM-backed arena and 25,890-param
// CNN don't fit M4's much tighter RAM budget; SDRAM is also normally
// M7-managed in this dual-core split, not something to improvise around
// here). M4 has no direct USB Serial -- all results relay to M7 via RPC,
// same pattern as h7_drongonet/drongonet_m4 and h7_goertzel_rpc. INA219
// energy reporting lives on M7 (that's where the sensor is wired and
// where it was already proven working in h7_drongonet_m7_instrumented);
// M4 just sends timing + score, M7 combines with its own power reading.

#include "RPC.h"
#include <arm_math.h>  // CMSIS-DSP, bundled with the Portenta board package

#ifdef abs
#undef abs
#endif

#include "Chirale_TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mel_tables.h"        // 16-mel, shared by Nano + Micro
#include "drongonet-nano.h"
#include "drongonet-micro.h"

// DWT cycle-counter macros -- see h7_drongonet_m7_instrumented for why
// these are inlined here rather than pulled from ARGUS_Common.h (that
// header's argus_print_system_storage() helper conflicts with this
// snap's installed core's FlashIAP.h declarations of __etext etc.).
#define DWT_ENABLE()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; \
                           DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while(0)
#define DWT_RESET()   do { DWT->CYCCNT = 0; } while(0)
#define DWT_CYCLES()  (DWT->CYCCNT)
#define DWT_US(cyc)   ((float)(cyc) / 480.0f)   // us at 480 MHz -- both cores run at the same clock here

// ==================== Model Dimensions ====================
const int MEL_BINS    = 16;
const int TIME_FRAMES = 184;
const int INPUT_SIZE  = MEL_BINS * TIME_FRAMES;  // 2944 elements

// ==================== Audio Settings ====================
const int micPin            = A0;  // matches this bench's actual mic wiring
const int SAMPLE_RATE       = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // 96 000 B BSS

#define RMS_THRESHOLD 500.0f

// Only Nano has a calibrated threshold (see h7_drongonet_m7_instrumented's
// file header note) -- 0.8, tuned against this room's aircon-hum baseline.
#define BIRD_SCORE_THRESHOLD 0.8f

// Which model this build measures. Run ONE at a time and reflash between
// them: M4's RAM is tight enough that two live interpreters hung the board
// outright (see the shared-arena note below), and measuring one model per
// flash also keeps each run's timings free of any residual effect from the
// other model's arena teardown.
#define RUN_MODEL_NANO  0
#define RUN_MODEL_MICRO 1

// Characterization run: average mel/infer/total latency over a fixed
// number of gated (RMS-passing) inferences, then report the average once
// and stop feeding further audio to the model -- this bench is meant for
// a short live-mic characterization run, not an unbounded log.
#define CHARACTERIZE_COUNT 100

// Q15 mel shadow-comparison run: times the fixed-point mel path against
// the trusted f32 one and reports an error metric, on the same real
// captured clip -- no model inference in this build (see RUN_MODELS
// below). Tried and NOT adopted -- see DRONGONET_SPARSE_MEL_BENCHMARK.md
// ("Q15 fixed-point FFT on M4"): arm_rfft_q15 came out both slower
// (~15%; this CMSIS-DSP version has no arm_rfft_fast_q15, only F32/F64
// got the optimized "fast" RFFT) and less accurate on real mic audio
// (fixed downscale-per-stage schedule loses much more precision on
// quiet real signals than it did on loud synthetic test tones) than the
// existing fastlog f32 path. Left in place for reference, off by
// default. When on, the 40 KB TFLite arena and both model interpreters
// are compiled out entirely (see the #if guards below and around the
// tensorArena declaration) rather than just left unused, because this
// sketch already has a documented history of M4 hanging silently at
// tight-but-compiler-reported-OK RAM margins (see the shared-arena note
// below) -- the Q15 buffers alone push well past that margin if the
// arena stays allocated alongside them.
#define RUN_Q15_COMPARE 0
#define RUN_MODELS ((RUN_MODEL_NANO || RUN_MODEL_MICRO) && !RUN_Q15_COMPARE)

// Sparse mel filterbank -- same trimming as h7_drongonet_m7_instrumented:
// drops each triangular filter's edge weights below threshold*peak
// (peak=1.0) instead of storing/accumulating the full startBin..endBin
// range. ~10% less filterbank work at threshold=0.1 for this 16-mel
// table. Only affects the mel-stage timing, not inference.
#define USE_SPARSE_MEL 1
#define SPARSE_MEL_THRESHOLD 0.1f

// ==================== TensorFlow Lite -- ONE shared arena ====================
// Holding two live interpreters at once (2 x 30 KB arenas) did not survive
// on M4: the sketch built fine at 85% RAM / ~42 KB stack headroom but hung
// before producing any output, while the identical RPC plumbing and an
// otherwise-blank M4 both worked -- consistent with h7_drongonet's own
// finding that M4 hangs silently past AllocateTensors() when RAM gets
// tight (a 64 KB single-model arena did it there; 40 KB was fine).
//
// So: one 40 KB arena, reused. Only one interpreter is alive at a time --
// each is constructed in its own scope, run, and torn down before the next
// model is built in the same memory. tflite::GetModel() just points into
// flash, so keeping both model handles costs nothing. Re-running
// AllocateTensors() each iteration costs a few ms but sits OUTSIDE the
// DWT-timed mel/inference regions, so it doesn't distort the measurement
// this sketch exists to take.
#if RUN_MODELS
namespace {
const tflite::Model *modelNano  = nullptr;
const tflite::Model *modelMicro = nullptr;

constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];
}  // namespace

// Micro characterization state (see CHARACTERIZE_COUNT above).
static int      microCharacterizeCount     = 0;
static uint32_t microCharacterizeSumMelUs  = 0;
static uint32_t microCharacterizeSumInferUs = 0;
static bool     microCharacterizeDone      = false;
#endif

// ==================== CMSIS-DSP FFT instance + buffers ====================
// NOTE: no .dtcmram section placement here -- unlike the M7-only
// instrumented sketch, DTCM overlaps RPC/OpenAMP's reserved region on M4
// and fails to link (see h7_drongonet/drongonet_m4.ino for the same
// finding). Plain SRAM is fine functionally, just a bit slower.
// Fast log10f approximation -- same technique as ARGUS_Common.h's
// argus_fast_log10f() (see h7_drongonet_m7_instrumented.ino's comment for
// why it's inlined here rather than including that header). ~10 cycles
// vs ~150 for libm log10f(), called 2944 times per inference -- the
// biggest lever on mel-stage time on this session's own measurements.
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

static arm_rfft_fast_instance_f32 fftInstance;
static float hannWindow[MEL_N_FFT];
static float fftBuf[MEL_N_FFT];
static float powerSpectrum[MEL_N_FFT_BINS];

// ==================== Q15 fixed-point mel path (shadow comparison) =====
// Cortex-M4 has the Thumb DSP-extension SIMD MACs that CMSIS-DSP's q15
// functions exploit (dual 16-bit MACs/cycle) -- unlike the f32 path,
// which just uses the single-precision FPU one value at a time. Audio is
// already int16 PCM, so q15 (-32768..32767 representing -1..~1) is a
// natural fit, not an artificial downcast.
//
// arm_rfft_q15's exact output packing/scaling isn't documented in prose
// in the CMSIS-DSP header (just an image reference), and this board uses
// the ARM_MATH_DSP-optimized asm bit-reversal path, not the generic C
// one visible in source -- so it was verified empirically on real
// hardware first (q15_rfft_diag/, kept for reference) rather than
// assumed from reading the source. Confirmed for fftLenReal=1024:
//   - pDst has length 2*fftLenReal and holds the FULL N-point complex
//     spectrum (not just the N/2+1 unique bins): pDst[2k]=Re(X[k]),
//     pDst[2k+1]=Im(X[k]), k=0..N-1. Only k=0..512 are needed (Hermitian
//     mirror for k=513..1023 confirmed bit-for-bit against k=511..1,
//     e.g. dst[1920]==dst[128] and dst[1918]==dst[126]==dst[1922] for a
//     bin-64 test tone). DC (k=0) and Nyquist (k=512) are purely real.
//   - Scale factor vs. arm_rfft_fast_f32's raw (unnormalized) output is
//     exactly N=1024 (bufF32[1]/dst[1024] = 4096000/4000 = 1024.000
//     exactly on a Nyquist-tone test case; other bins landed within
//     ~0.3% of that, attributable to ordinary q15 quantization noise).
//     Not corrected for here: a constant scale on power is a constant
//     additive shift in the log domain (10*log10(c*x) = 10*log10(c) +
//     10*log10(x)), which the existing per-clip global min-max
//     normalize step removes completely, same reasoning already applies
//     to the f32 path's own arbitrary unnormalized scale.
//
// Buffers (~10 KB) gated behind RUN_Q15_COMPARE so the default
// model-running build doesn't pay for them -- this sketch's RAM margin
// is already tight enough that unused-but-allocated static buffers are
// not free here (see the shared-arena note further down).
#if RUN_Q15_COMPARE
static arm_rfft_instance_q15 fftInstanceQ15;
static q15_t hannWindowQ15[MEL_N_FFT];
static q15_t audioFrameQ15[MEL_N_FFT];   // pSrc -- arm_rfft_q15 modifies this in place
static q15_t fftDstQ15[2 * MEL_N_FFT];
#endif

// ==================== Flat mel filterbank weight table ====================
#define FLAT_WEIGHT_TOTAL 931

typedef struct {
  int startBin;
  int width;
  int offset;
} MelFilterFlat;

static float melWeightFlat[FLAT_WEIGHT_TOTAL];
#if RUN_Q15_COMPARE
static q15_t melWeightFlatQ15[FLAT_WEIGHT_TOTAL];
#endif
static MelFilterFlat melFiltersFlat[MEL_N_MELS];

#if RUN_MODELS
// ===========================================================
// Quantization param helper -- Micro has genuine INT8 graph I/O (see
// h7_drongonet_m7_instrumented's file header note for the full Nano vs
// Micro/Edge dtype story). Nano's I/O is FLOAT32, no helper needed there.
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

// ===========================================================
// Per-model runners -- each builds its interpreter in the shared arena,
// runs, reports, and tears down before returning, so only one is ever
// live (see the shared-arena note above).
// ===========================================================

// Nano: FLOAT32 graph I/O -- write mel floats straight in, read P(bird)
// straight out. 8-op resolver (explicit QUANTIZE/DEQUANTIZE boundary ops).
void runNano(const float *melFeatures, int rms, int melUsInt) {
  static tflite::MicroMutableOpResolver<8> resolver;
  static bool resolverInit = false;
  if (!resolverInit) {
    resolver.AddQuantize();
    resolver.AddMul();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddDequantize();
    resolverInit = true;
  }

  tflite::MicroInterpreter interpreter(modelNano, resolver, tensorArena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    RPC.call("reportError", "nano AllocateTensors failed");
    return;
  }

  TfLiteTensor *input  = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);

  memcpy(input->data.f, melFeatures, INPUT_SIZE * sizeof(float));

  DWT_RESET();
  TfLiteStatus status = interpreter.Invoke();
  uint32_t inferCycles = DWT_CYCLES();
  float inferUs = DWT_US(inferCycles);

  if (status != kTfLiteOk) {
    RPC.call("reportError", "nano invoke failed");
    return;
  }

  float probBird = output->data.f[1];
  RPC.call("reportNano", rms, (int)(probBird * 1000.0f),
           melUsInt, (int)(inferUs + 0.5f));
  if (probBird > BIRD_SCORE_THRESHOLD) {
    RPC.call("handleBird", "BIRD");
  }
}

// Micro: genuine INT8 graph I/O -- real quantize/dequantize against the
// tensors' own scale/zero_point. Folded graph, 6 ops, no
// QUANTIZE/DEQUANTIZE nodes.
void runMicro(const float *melFeatures, int rms, int melUsInt) {
  static tflite::MicroMutableOpResolver<6> resolver;
  static bool resolverInit = false;
  if (!resolverInit) {
    resolver.AddMul();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolverInit = true;
  }

  tflite::MicroInterpreter interpreter(modelMicro, resolver, tensorArena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    RPC.call("reportError", "micro AllocateTensors failed");
    return;
  }

  TfLiteTensor *input  = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);

  float   inScale, outScale;
  int32_t inZp, outZp;
  getTensorQuantParams(input,  &inScale,  &inZp);
  getTensorQuantParams(output, &outScale, &outZp);

  for (int i = 0; i < INPUT_SIZE; i++) {
    float scaled = melFeatures[i] / inScale + inZp;
    if (scaled >  127.0f) scaled =  127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    input->data.int8[i] = (int8_t)scaled;
  }

  DWT_RESET();
  TfLiteStatus status = interpreter.Invoke();
  uint32_t inferCycles = DWT_CYCLES();
  float inferUs = DWT_US(inferCycles);

  if (status != kTfLiteOk) {
    RPC.call("reportError", "micro invoke failed");
    return;
  }

  float probBird = (output->data.int8[1] - outZp) * outScale;
  int inferUsInt = (int)(inferUs + 0.5f);
  RPC.call("reportMicro", rms, (int)(probBird * 1000.0f),
           melUsInt, inferUsInt);

  microCharacterizeSumMelUs   += (uint32_t)melUsInt;
  microCharacterizeSumInferUs += (uint32_t)inferUsInt;
  microCharacterizeCount++;
  if (microCharacterizeCount >= CHARACTERIZE_COUNT && !microCharacterizeDone) {
    microCharacterizeDone = true;
    float avgMel   = (float)microCharacterizeSumMelUs   / microCharacterizeCount;
    float avgInfer = (float)microCharacterizeSumInferUs / microCharacterizeCount;
    float avgTotal = avgMel + avgInfer;
    RPC.call("reportMicroAverages", microCharacterizeCount,
             (int)(avgMel + 0.5f), (int)(avgInfer + 0.5f), (int)(avgTotal + 0.5f));
  }
}
#endif  // RUN_MODELS

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

  int offset = 0;
  for (int m = 0; m < MEL_N_MELS; m++) {
    int s = melFilters[m].startBin;
    int p = melFilters[m].peakBin;
    int e = melFilters[m].endBin;
    float rise_inv = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv = (e > p) ? 1.0f / (float)(e - p) : 0.0f;

#if USE_SPARSE_MEL
    while (s < p) {
      float w = (float)(s - melFilters[m].startBin) * rise_inv;
      if (w >= SPARSE_MEL_THRESHOLD) break;
      s++;
    }
    while (e > p) {
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
#if RUN_Q15_COMPARE
      // Q15 weight: same triangle shape, but NOT pre-scaled by 1/32768 --
      // arm_mult_q15's own (a*b)>>15 already treats this operand as a
      // 0..1 fraction, so audio stays on its native raw-int16 scale
      // after windowing, matching how the f32 path (pre-scaled window)
      // lands on the same effective scale via a different route.
      melWeightFlatQ15[offset + (k - s)] = (q15_t)(w * 32767.0f);
#endif
    }
    offset += width;
  }

#if RUN_Q15_COMPARE
  // Q15 Hann window -- deliberately NOT pre-scaled by 1/32768 (see the
  // f32 window's comment above for why that scaling exists there); q15
  // arithmetic gets the equivalent effect for free from arm_mult_q15's
  // own >>15, since raw int16 audio samples are already Q15-format
  // values on the -1..~1 scale.
  for (int i = 0; i < MEL_N_FFT; i++) {
    float w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)MEL_N_FFT));
    hannWindowQ15[i] = (q15_t)(w * 32767.0f);
  }
  arm_rfft_init_q15(&fftInstanceQ15, MEL_N_FFT, 0 /*forward*/, 1 /*normal order*/);
#endif
}

// ===========================================================
// Mel Spectrogram -- FFT-based, matching training pipeline
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

#if RUN_Q15_COMPARE
// ===========================================================
// Mel Spectrogram -- Q15 fixed-point FFT path (shadow comparison only;
// not fed to either model here). See the buffer-declaration comment
// above for the packing/scale story this relies on. Tried and NOT
// adopted -- see the RUN_Q15_COMPARE definition's comment and
// DRONGONET_SPARSE_MEL_BENCHMARK.md for why (slower + less accurate on
// real audio than the existing fastlog f32 path). Kept for reference.
// ===========================================================
void computeMelSpectrogramQ15(int16_t *audio, int audioLength, float *melOut) {
  static int32_t powerSpectrumQ15[MEL_N_FFT_BINS];  // Re^2+Im^2, up to ~2^31 -- int32 is enough headroom

  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart   = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    const int16_t *src = &audio[frameStart];
    if (validSamples < MEL_N_FFT) {
      memcpy(audioFrameQ15, src, validSamples * sizeof(int16_t));
      memset(&audioFrameQ15[validSamples], 0, (MEL_N_FFT - validSamples) * sizeof(int16_t));
    } else {
      memcpy(audioFrameQ15, src, MEL_N_FFT * sizeof(int16_t));
    }
    // arm_mult_q15: out = saturate((a*b)>>15) -- window treated as a
    // 0..1 Q15 fraction, audio stays on its native raw-int16 scale.
    arm_mult_q15(audioFrameQ15, hannWindowQ15, audioFrameQ15, MEL_N_FFT);

    // pSrc (audioFrameQ15) is modified as scratch by this call; pDst
    // (fftDstQ15) ends up holding the full N-point complex spectrum,
    // pDst[2k]=Re(X[k]), pDst[2k+1]=Im(X[k]), k=0..N-1 (verified on
    // real hardware, see the declaration comment above).
    arm_rfft_q15(&fftInstanceQ15, audioFrameQ15, fftDstQ15);

    // DC and Nyquist are purely real (packed at index 0 and N respectively).
    int32_t dc = fftDstQ15[0];
    powerSpectrumQ15[0] = dc * dc;
    int32_t ny = fftDstQ15[MEL_N_FFT];
    powerSpectrumQ15[MEL_N_FFT / 2] = ny * ny;
    for (int k = 1; k < MEL_N_FFT / 2; k++) {
      int32_t re = fftDstQ15[2 * k];
      int32_t im = fftDstQ15[2 * k + 1];
      powerSpectrumQ15[k] = re * re + im * im;
    }

    float *melRow = &melOut[f * MEL_N_MELS];
    for (int m = 0; m < MEL_N_MELS; m++) {
      const int32_t *P = &powerSpectrumQ15[melFiltersFlat[m].startBin];
      const q15_t   *W = &melWeightFlatQ15[melFiltersFlat[m].offset];
      int width = melFiltersFlat[m].width;

      // int64 accumulator: each term is up to ~2^31 (power) x ~2^15
      // (q15 weight) = ~2^46, and up to ~58 terms typically -- nowhere
      // near int64 range, but well past int32.
      int64_t acc = 0;
      for (int k = 0; k < width; k++) {
        acc += (int64_t)P[k] * (int64_t)W[k];
      }

      // No N-scale correction applied here (see declaration comment --
      // a constant multiplicative scale becomes a constant additive
      // shift in the log domain, removed by the min-max normalize
      // below, same as the f32 path's own unnormalized scale).
      melRow[m] = 10.0f * fastLog10f((float)acc + 1e-10f);
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
#endif  // RUN_Q15_COMPARE

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
  pinMode(micPin, INPUT);
  analogReadResolution(12);  // this board defaults to 10-bit (idle ~512) --
                             // captureAudio()'s -2048/2048 scaling assumes
                             // true 12-bit (0-4095, idle ~2048). Same fix
                             // as h7_drongonet_m7_instrumented; goertzel_m4
                             // independently confirmed the 10-bit default.
  RPC.begin();

  DWT_ENABLE();

#if RUN_MODELS
  // Model handles only -- tflite::GetModel() just points into flash. The
  // interpreters themselves are built per-run inside runNano()/runMicro()
  // so only one occupies the shared arena at a time.
  modelNano = tflite::GetModel(drongonet_nano);
  if (!modelNano || modelNano->version() != TFLITE_SCHEMA_VERSION) {
    while (1) { }  // no Serial on M4 -- nothing to recover to without a working model
  }

  modelMicro = tflite::GetModel(drongonet_micro);
  if (!modelMicro || modelMicro->version() != TFLITE_SCHEMA_VERSION) {
    while (1) { }
  }
#endif

  initMelFFT();

  // M7 boots M4 from inside its own RPC.begin(), then binds its handlers.
  // Calling before those binds land races the handshake (RPC.call blocks
  // waiting for a reply that no bound function will produce), so give M7 a
  // moment -- 2 s was confirmed sufficient with an RPC-only test build.
  delay(2000);
#if RUN_MODELS
  RPC.call("reportReady", (int)kTensorArenaSize, (int)kTensorArenaSize);
#else
  RPC.call("reportReady", 0, 0);
#endif
}

void loop() {
#if RUN_MODEL_MICRO
  if (microCharacterizeDone) {
    delay(5000);  // characterization complete -- idle, don't keep capturing/inferring
    return;
  }
#endif

  captureAudio();

  float rmsAccum = 0.0f;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float s = (float)audioBuffer[i];
    rmsAccum += s * s;
  }
  float rms = sqrtf(rmsAccum / (float)BUFFER_SIZE);

  if (rms < RMS_THRESHOLD) {
    RPC.call("reportGate", (int)rms);
    return;
  }

  static float melFeatures[INPUT_SIZE];

  // Mel is computed once and reused by both models (identical 16-mel
  // filterbank), so the mel figure they each report is the same number by
  // construction, not a coincidence.
  DWT_RESET();
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, melFeatures);
  uint32_t melCycles = DWT_CYCLES();
  float melUs = DWT_US(melCycles);

  // Compressed payload: fixed-point ints instead of floats -- same
  // (int)(score*1000) convention h7_goertzel_rpc/drongonet_m4.ino already
  // use for reportScore/reportMagnitude. msgpack encodes small ints in
  // 1-3 bytes vs 5 for a float32, so this is a real reduction in what
  // crosses the RPC/OpenAMP link per report, not just a style choice.
  int rmsInt   = (int)rms;
  int melUsInt = (int)(melUs + 0.5f);

#if RUN_Q15_COMPARE
  // Shadow comparison only -- q15 mel features are NOT fed to either
  // model here, just timed and compared against the trusted f32 output
  // on this same real captured clip, on real hardware.
  static float melFeaturesQ15[INPUT_SIZE];
  DWT_RESET();
  computeMelSpectrogramQ15(audioBuffer, BUFFER_SIZE, melFeaturesQ15);
  uint32_t melCyclesQ15 = DWT_CYCLES();
  float melUsQ15 = DWT_US(melCyclesQ15);

  float maxAbsDiff = 0.0f, sumAbsDiff = 0.0f;
  for (int i = 0; i < INPUT_SIZE; i++) {
    float d = fabsf(melFeatures[i] - melFeaturesQ15[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
    sumAbsDiff += d;
  }
  float meanAbsDiff = sumAbsDiff / (float)INPUT_SIZE;

  RPC.call("reportQ15Compare", rmsInt, melUsInt, (int)(melUsQ15 + 0.5f),
           (int)(maxAbsDiff * 1000.0f), (int)(meanAbsDiff * 1000.0f));
#endif

#if RUN_MODELS
#if RUN_MODEL_NANO
  runNano(melFeatures, rmsInt, melUsInt);
#endif
#if RUN_MODEL_MICRO
  runMicro(melFeatures, rmsInt, melUsInt);
#endif
#endif
}
