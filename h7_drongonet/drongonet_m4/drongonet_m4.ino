// drongonet_m4.ino — Tier-1 CNN bird-activity gatekeeper on M4, RPC to M7
// Model: DrongoNet-nano (INT8, 16x184 mel spectrogram -> softmax[no_bird,bird])
// Same DSP/inference pipeline as src/DrongoNet/DrongoNet_Nano, stripped of
// its INA219/DWT/ARGUS_Common benchmarking scaffolding and pointed at RPC
// instead of Serial, so it can run standalone on M4 and wake M7 like
// h7_goertzel_rpc does -- see h7_goertzel_rpc/README.md for why this pairing
// exists (proving the M4-gatekeeper/M7-classifier RPC pattern
// HOW_ARGUS_WORKS.md describes, decoupled from any model choice).

#include "RPC.h"
#include <arm_math.h>  // CMSIS-DSP, bundled with the Portenta board package

#ifdef abs
#undef abs
#endif

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mel_tables.h"
#include "drongonet-nano.h"

// ==================== Model Dimensions ====================
const int MEL_BINS    = 16;
const int TIME_FRAMES = 184;
const int INPUT_SIZE  = MEL_BINS * TIME_FRAMES;  // 2944 elements

// ==================== Audio Settings ====================
const int micPin            = A1;  // matches src/DrongoNet/DrongoNet_Nano wiring
const int SAMPLE_RATE       = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // 96 000 B BSS

// RMS energy gate, ADC units (int16 scale +-32767) -- skip inference on
// silent windows. Same starting point as DrongoNet_Nano; recalibrate for
// your mic/gain.
#define RMS_THRESHOLD 500.0f

// avgScore > this -> bird detected. Same default as DrongoNet_Nano.
#define BIRD_SCORE_THRESHOLD 0.6f

// ==================== TensorFlow Lite ====================
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model            = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input  = nullptr;
TfLiteTensor *output = nullptr;

// DrongoNet-nano typically uses ~28-40 KB of arena (see
// src/DrongoNet/DrongoNet_Nano's profiling comments). 64 KB (matching that
// sketch's headroom) hangs M4 silently somewhere past AllocateTensors() --
// M4's RAM budget is far smaller than M7's, and between the 96 KB audio
// buffer and everything else, 64 KB of arena apparently leaves too little
// heap/stack margin for the runtime to survive. 40 KB (still ~1.3x the
// documented typical usage) is confirmed working; don't bump this back up
// without re-verifying against a real hang.
constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ==================== CMSIS-DSP FFT instance + buffers ====================
static arm_rfft_fast_instance_f32 fftInstance;

// Note: no explicit .dtcmram placement here (unlike the M7-only
// src/DrongoNet/DrongoNet_Nano) -- that section overlaps RPC/OpenAMP's
// reserved DTCM region on M4 and fails to link. Plain SRAM is fine
// functionally, just a bit slower.
static float hannWindow[MEL_N_FFT];
static float fftBuf[MEL_N_FFT];
static float powerSpectrum[MEL_N_FFT_BINS];

// ==================== Flat mel filterbank weight table ====================
#define FLAT_WEIGHT_TOTAL 931

typedef struct {
  int startBin;
  int width;
  int offset;
} MelFilterFlat;

static float melWeightFlat[FLAT_WEIGHT_TOTAL];
static MelFilterFlat melFiltersFlat[MEL_N_MELS];

// 3-window ring buffer for temporal confidence smoothing -- a single noisy
// frame cannot trigger detection.
static float scoreHistory[3] = {0.0f, 0.0f, 0.0f};
static int scoreHistIdx = 0;
static bool birdDetected = false;

// ===========================================================
// CMSIS-DSP FFT + Hann window + flat filterbank initialisation
// ===========================================================
void initMelFFT() {
  for (int i = 0; i < MEL_N_FFT; i++) {
    hannWindow[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)MEL_N_FFT));
  }

  arm_rfft_fast_init_f32(&fftInstance, MEL_N_FFT);

  int offset = 0;
  for (int m = 0; m < MEL_N_MELS; m++) {
    int s = melFilters[m].startBin;
    int p = melFilters[m].peakBin;
    int e = melFilters[m].endBin;
    int width = e - s;
    melFiltersFlat[m].startBin = s;
    melFiltersFlat[m].width    = width;
    melFiltersFlat[m].offset   = offset;
    float rise_inv = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv = (e > p) ? 1.0f / (float)(e - p) : 0.0f;
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
// ~150 for libm log10f(), error < 0.5 dB, negligible after the per-clip
// min-max normalize below). Same technique as ARGUS_Common.h's
// argus_fast_log10f() / h7_drongonet_m4_instrumented.ino's fastLog10f(),
// inlined here rather than #include-ing ARGUS_Common.h (that header
// pulls in <mbed.h>/<mbed_stats.h> unconditionally, which conflicts with
// this M4-side sketch's own includes -- see
// h7_drongonet_m4_instrumented.ino's file header for the same finding).
// Confirmed ~16% mel-stage win on M4 in DRONGONET_SPARSE_MEL_BENCHMARK.md
// -- ported here to close the gap between this earlier dev sketch and
// that finding.
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

// ===========================================================
// Mel Spectrogram -- FFT-based, matching training pipeline
// ===========================================================
void computeMelSpectrogram(int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart   = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    const float INV_32768 = 1.0f / 32768.0f;
    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) {
      fftBuf[i] = (float)src[i] * (hannWindow[i] * INV_32768);
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
  RPC.begin();

  model = tflite::GetModel(drongonet_nano);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
    while (1) {
    }  // model load / schema mismatch -- nothing to recover to without Serial
  }

  // DrongoNet-nano exact op set -- see src/DrongoNet/DrongoNet_Nano for the
  // decoded-from-flatbuffer op list this mirrors.
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddQuantize();
  resolver.AddMul();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    while (1) {
    }  // arena too small -- bump kTensorArenaSize
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  initMelFFT();
}

void loop() {
  captureAudio();

  float rmsAccum = 0.0f;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float s = (float)audioBuffer[i];
    rmsAccum += s * s;
  }
  float rms = sqrtf(rmsAccum / (float)BUFFER_SIZE);

  if (rms < RMS_THRESHOLD) {
    return;  // silent window, skip inference
  }

  static float melFeatures[INPUT_SIZE];
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, melFeatures);

  for (int i = 0; i < INPUT_SIZE; i++) {
    float scaled = melFeatures[i] / input->params.scale + input->params.zero_point;
    if (scaled >  127.0f) scaled =  127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    input->data.int8[i] = (int8_t)scaled;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    return;
  }

  // nano uses Dense(2, softmax): output[0]=P(no_bird), output[1]=P(bird)
  float prob_bird = (output->data.int8[1] - output->params.zero_point) * output->params.scale;

  scoreHistory[scoreHistIdx % 3] = prob_bird;
  scoreHistIdx++;
  float avgScore = (scoreHistory[0] + scoreHistory[1] + scoreHistory[2]) / 3.0f;

  // Live score report every inference cycle (naturally rate-limited to
  // ~once per capture window, ~3s+) for threshold calibration -- same
  // "measure before guessing" pattern as h7_goertzel_rpc's reportMagnitude.
  RPC.call("reportScore", (int)(avgScore * 1000));

  bool detected = (avgScore > BIRD_SCORE_THRESHOLD);
  if (detected && !birdDetected) {
    birdDetected = true;
    RPC.call("handleBird", "BIRD");
  } else if (!detected) {
    birdDetected = false;
  }
}
