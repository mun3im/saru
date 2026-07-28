#include <Arduino.h>
#include "SDRAM.h"
#include <arm_math.h>

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Shared ARGUS helpers: DWT timing macros, fast_log10f, StageRAM profiling,
// and the system storage breakdown -- see
// libraries/ARGUS_Common/src/ARGUS_Common.h. Install this library (copy
// libraries/ARGUS_Common into your Arduino libraries folder, or open the
// .ino via the repo checkout) before compiling this sketch.
#include <ARGUS_Common.h>

// Include INA219 Power Monitor
#include <Adafruit_INA219.h>
#include <Wire.h>

Adafruit_INA219 ina219;
bool ina219_ok = false;
float total_energy_mWh = 0.0f;
unsigned long lastPowerMs = 0;

// Include the models
#include "drongonet-micro.h"
#include "Mynanet_1j.h"

// Include Mel Tables — MyDrongo (16-bin, 1024-pt FFT)
#include "mel_tables.h"

// Include Mel Tables — MynaNet (64-bin, 512-pt FFT)
#include "mynanet_mel_tables_64.h"

// ── Audio Configuration ───────────────────────────────────────────────────────
const int micPin           = A1;
const int SAMPLE_RATE      = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE      = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000; // 48 000

// ── Global audio buffer ────────────────────────────────────────────────────────
// 48 000 × 2 B = 96 000 B = 93.75 KB  (BSS / zero-init static RAM)
int16_t audioBuffer[BUFFER_SIZE];

// ── MyDrongo (DrongoNet-micro) feature dimensions ────────────────────────────────
const int MEL_BINS    = 16;
const int TIME_FRAMES = 184;
const int INPUT_SIZE  = MEL_BINS * TIME_FRAMES; // 2 944

// MyDrongo feature buffer — float normalised [0,1] fed into the quantiser
float bad_features[INPUT_SIZE];

// ── CMSIS-DSP FFT instance + buffers for MyDrongo (DTCM placement) ───────────────
// DTCM is zero wait-state on STM32H7; placing the FFT working set here
// eliminates AXI SRAM cache-miss stalls (~3–8 ms saving for 184-frame run).
// NOTE: Standard DMA cannot access DTCM.  captureAudio() uses CPU-polled ADC,
// so this placement is safe.
static arm_rfft_fast_instance_f32 fftInstance;

__attribute__((section(".dtcmram")))
static float hannWindow[MEL_N_FFT];         // 1024 floats = 4 KB → DTCM

__attribute__((section(".dtcmram")))
static float fftBuf[MEL_N_FFT];            // 1024 floats = 4 KB → DTCM (in/out)

__attribute__((section(".dtcmram")))
static float powerSpectrum[MEL_N_FFT_BINS]; // 513 floats = 2 KB → DTCM

// MyDrongo flat mel filterbank ───────────────────────────────────────────────────
// Total non-zero bins across all 16 filters = 931 → 3.6 KB BSS
#define FLAT_WEIGHT_TOTAL 931
typedef struct {
  int startBin;
  int width;
  int offset;
} MelFilterFlat;

static float       melWeightFlat[FLAT_WEIGHT_TOTAL];
static MelFilterFlat melFiltersFlat[MEL_N_MELS];

// ── MynaNet (DS-CNN) Mel spectrogram — 64×300 ────────────────────────────────
// Audio parameters: sr=16 kHz, n_fft=512, hop_length=160, n_mels=64, 3 s clip.
// FFT instance and working buffers are kept separate from the MyDrongo buffers to
// avoid any interference; both sets are in BSS (not DTCM) because DTCM is
// already filled by the MyDrongo windows.
//
// MYNANET_N_FFT_BINS = 257  (512/2 + 1)
// MYNANET_INPUT_SIZE = 64 × 300 = 19 200 floats = 75 KB (allocated in SDRAM)

static arm_rfft_fast_instance_f32 fftInstance_Myna;

// Hann window for the 512-pt MynaNet FFT
static float hannWindow_Myna[MYNANET_N_FFT];          // 512 floats = 2 KB

// Shared scratch FFT buffer for MynaNet (reuses same working buffer pattern)
static float fftBuf_Myna[MYNANET_N_FFT];             // 512 floats = 2 KB
static float powerSpectrum_Myna[MYNANET_N_FFT_BINS]; // 257 floats = 1 KB

// Flat filterbank weights for MynaNet 64-mel filters.
// Upper bound on total non-zero bins for n_fft=512, n_mels=64, fmin=0, fmax=8000:
// computed at runtime in initMelFFT_Myna(); max possible = 64×256 ≈ 4 KB safety margin.
// In practice the triangular filters for these params total ~1 600 non-zero weights.
#define MYNA_FLAT_WEIGHT_TOTAL 4096

static float           mynanetMelWeightFlat[MYNA_FLAT_WEIGHT_TOTAL];
static MynanetMelFilterFlat mynanetMelFiltersFlat[MYNANET_N_MELS];

// MynaNet feature output buffer — 64 × 300 = 19 200 floats = 75 KB.
// This is too large for DTCM or internal SRAM; pointer lives in SDRAM.
// Allocated just after the MynaNet tensor arena in SDRAM (see setup()).
float *myna_features = nullptr;

// ── TFLite Globals ────────────────────────────────────────────────────────────
// MyDrongo Model (DrongoNet-micro)
const tflite::Model   *bad_model       = nullptr;
tflite::MicroInterpreter *bad_interpreter = nullptr;
TfLiteTensor          *bad_input       = nullptr;
TfLiteTensor          *bad_output      = nullptr;

// MynaNet Model
const tflite::Model   *myna_model       = nullptr;
tflite::MicroInterpreter *myna_interpreter = nullptr;
TfLiteTensor          *myna_input       = nullptr;
TfLiteTensor          *myna_output      = nullptr;

// ── Tensor arenas ─────────────────────────────────────────────────────────────
constexpr int kBadArenaSize  = 150 * 1024;
alignas(16) uint8_t bad_tensor_arena[kBadArenaSize];

constexpr int kMynaArenaSize = 1500 * 1024;
uint8_t *myna_tensor_arena   = nullptr;  // allocated in SDRAM

// ── Temporal smoothing (3-window ring buffer) ─────────────────────────────────
// Matches MyDrongo approach: a single noisy frame cannot alone trigger a detection.
static float scoreHistory[3]  = {0.0f, 0.0f, 0.0f};
static int   scoreHistIdx     = 0;

// fast_log10f now lives in ARGUS_Common.h as argus_fast_log10f() -- see
// the #include near the top of this file.

// ──────────────────────────────────────────────────────────────────────────────
// initMelFFT — MyDrongo  (16-mel, 1024-pt RFFT)
// ──────────────────────────────────────────────────────────────────────────────
// Builds the Hann window, initialises the CMSIS RFFT instance, and pre-bakes
// the triangular filterbank weights into the flat table so the per-frame hot
// path contains no divisions.  Matches MyDrongo.ino § initMelFFT().
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
      if      (k < p)  w = (float)(k - s) * rise_inv;
      else if (k == p) w = 1.0f;
      else             w = (float)(e - k) * fall_inv;
      melWeightFlat[offset + (k - s)] = w;
    }
    offset += width;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// initMelFFT_Myna — MynaNet  (64-mel, 512-pt RFFT)
// ──────────────────────────────────────────────────────────────────────────────
void initMelFFT_Myna() {
  // ── Hann window (512-pt zero-padded from 400-pt) ──────────────────────
  // librosa defaults to win_length = 400 when hop_length=160 and n_fft=512
  const int win_length = 400;
  const int pad_left = (MYNANET_N_FFT - win_length) / 2; // 56
  
  for (int i = 0; i < MYNANET_N_FFT; i++) {
    if (i >= pad_left && i < pad_left + win_length) {
      int w_i = i - pad_left;
      hannWindow_Myna[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * w_i / (float)win_length));
    } else {
      hannWindow_Myna[i] = 0.0f;
    }
  }

  // ── Initialise 512-pt CMSIS RFFT instance ────────────────────────────
  arm_rfft_fast_init_f32(&fftInstance_Myna, MYNANET_N_FFT);

  // ── Compute mel bin edge points from Slaney formula ──────────────────
  const int n_points = MYNANET_N_MELS + 2; // 66 edge points for 64 filters
  
  // Slaney mel conversions
  auto hz_to_mel_slaney = [](float hz) -> float {
      if (hz >= 1000.0f) {
          return 15.0f + logf(hz / 1000.0f) / (logf(6.4f) / 27.0f);
      } else {
          return hz * 3.0f / 200.0f;
      }
  };
  auto mel_to_hz_slaney = [](float mel) -> float {
      if (mel >= 15.0f) {
          return 1000.0f * expf((logf(6.4f) / 27.0f) * (mel - 15.0f));
      } else {
          return mel * 200.0f / 3.0f;
      }
  };

  float mel_min = hz_to_mel_slaney(MYNANET_FMIN);
  float mel_max = hz_to_mel_slaney(MYNANET_FMAX);
  float mel_step = (mel_max - mel_min) / (float)(n_points - 1);

  float hz_pts[n_points];
  for (int i = 0; i < n_points; i++) {
    float mel = mel_min + i * mel_step;
    hz_pts[i] = mel_to_hz_slaney(mel);
  }

  // ── Pre-bake flat filterbank weight table ─────────────────────────────
  // Computes exact librosa slaney-norm filter weights at each FFT bin.
  int offset = 0;
  for (int m = 0; m < MYNANET_N_MELS; m++) {
    float f_start = hz_pts[m];
    float f_peak  = hz_pts[m + 1];
    float f_end   = hz_pts[m + 2];
    
    // librosa norm='slaney' area normalization factor
    float enorm = 2.0f / (f_end - f_start);
    
    int first_bin = -1;
    int last_bin = -1;
    float temp_weights[MYNANET_N_FFT_BINS];
    
    for (int k = 0; k < MYNANET_N_FFT_BINS; k++) {
      float bin_hz = (float)k * (float)MYNANET_SR / (float)MYNANET_N_FFT;
      float w = 0.0f;
      if (bin_hz >= f_start && bin_hz <= f_peak && f_peak > f_start) {
        w = (bin_hz - f_start) / (f_peak - f_start);
      } else if (bin_hz > f_peak && bin_hz <= f_end && f_end > f_peak) {
        w = (f_end - bin_hz) / (f_end - f_peak);
      }
      
      w *= enorm;
      temp_weights[k] = w;
      
      if (w > 0.0f) {
        if (first_bin == -1) first_bin = k;
        last_bin = k;
      }
    }
    
    if (first_bin == -1) {
      // Degenerate filter — store one zero-weight bin as a placeholder
      mynanetMelFiltersFlat[m].startBin = 0;
      mynanetMelFiltersFlat[m].width    = 1;
      mynanetMelFiltersFlat[m].offset   = offset;
      mynanetMelWeightFlat[offset]      = 0.0f;
      offset += 1;
    } else {
      int width = last_bin - first_bin + 1;
      mynanetMelFiltersFlat[m].startBin = first_bin;
      mynanetMelFiltersFlat[m].width    = width;
      mynanetMelFiltersFlat[m].offset   = offset;
      
      for (int k = first_bin; k <= last_bin; k++) {
        mynanetMelWeightFlat[offset + (k - first_bin)] = temp_weights[k];
      }
      offset += width;
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// computeMelSpectrogram — MyDrongo  (16-mel, 1024-pt, 184 frames)
// ──────────────────────────────────────────────────────────────────────────────
// Reproduces the librosa chain used in DrongoNet-micro training:
//   melspectrogram(y, sr=16000, n_fft=1024, hop=256, n_mels=16, fmin=100, fmax=8000)
//   power_to_db → min-max normalise to [0,1]
//
// Optimisations (see MyDrongo.ino for full rationale):
//   OPT-A: Fused int16→float + Hann window (single pass, halves fftBuf write bandwidth)
//   OPT-B: Inlined 4-way unrolled filterbank dot-product (removes 2944 call overheads)
//   OPT-C: SIMD normalisation via arm_offset_f32 + arm_scale_f32
//   OPT-D: DTCM buffer placement (hannWindow, fftBuf, powerSpectrum)
void computeMelSpectrogram(int16_t *audio, int audioLength, float *melOut) {
  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart    = f * MEL_HOP_LENGTH;
    int validSamples  = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    // OPT-A: Fused convert + Hann window
    const float   INV_32768 = 1.0f / 32768.0f;
    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) {
      fftBuf[i] = (float)src[i] * (hannWindow[i] * INV_32768);
    }
    if (validSamples < MEL_N_FFT) {
      memset(&fftBuf[validSamples], 0, (MEL_N_FFT - validSamples) * sizeof(float));
    }

    arm_rfft_fast_f32(&fftInstance, fftBuf, fftBuf, 0);

    powerSpectrum[0]           = fftBuf[0] * fftBuf[0];
    powerSpectrum[MEL_N_FFT/2] = fftBuf[1] * fftBuf[1];
    arm_cmplx_mag_squared_f32(&fftBuf[2], &powerSpectrum[1], MEL_N_FFT / 2 - 1);

    // OPT-B: Inlined 4-way unrolled filterbank dot-product
    float *melRow = &melOut[f * MEL_N_MELS];
    for (int m = 0; m < MEL_N_MELS; m++) {
      const float *P     = &powerSpectrum[melFiltersFlat[m].startBin];
      const float *W     = &melWeightFlat[melFiltersFlat[m].offset];
      int          width =  melFiltersFlat[m].width;

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
      melRow[m] = 10.0f * argus_fast_log10f(acc + 1e-10f);
    }
  }

  // OPT-C: SIMD normalisation
  float globalMin, globalMax;
  uint32_t minIdx, maxIdx;
  arm_min_f32(melOut, INPUT_SIZE, &globalMin, &minIdx);
  arm_max_f32(melOut, INPUT_SIZE, &globalMax, &maxIdx);

  float range    = globalMax - globalMin;
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

// ──────────────────────────────────────────────────────────────────────────────
// computeMelSpectrogram_Myna — MynaNet  (64-mel, 512-pt, 300 frames)
// ──────────────────────────────────────────────────────────────────────────────
void computeMelSpectrogram_Myna(int16_t *audio, int audioLength, float *melOut) {
  const int n_mels_myna   = MYNANET_N_MELS;       // 64
  const int n_frames_myna = MYNANET_TIME_FRAMES;  // 300

  for (int f = 0; f < n_frames_myna; f++) {
    int center_idx = f * MYNANET_HOP_LENGTH;
    int start_idx  = center_idx - MYNANET_N_FFT / 2;

    // OPT-A: Fused int16→float + Hann window (512-pt)
    const float INV_32768 = 1.0f / 32768.0f;
    for (int i = 0; i < MYNANET_N_FFT; i++) {
      int sampleIdx = start_idx + i;
      int16_t sample = 0;
      if (sampleIdx >= 0 && sampleIdx < audioLength) {
        sample = audio[sampleIdx];
      } else {
        // librosa center=True default pad_mode is 'reflect'
        if (sampleIdx < 0) {
          sample = audio[-sampleIdx];
        } else if (sampleIdx >= audioLength) {
          sample = audio[2 * audioLength - 2 - sampleIdx];
        }
      }
      fftBuf_Myna[i] = (float)sample * (hannWindow_Myna[i] * INV_32768);
    }

    arm_rfft_fast_f32(&fftInstance_Myna, fftBuf_Myna, fftBuf_Myna, 0);

    powerSpectrum_Myna[0]                 = fftBuf_Myna[0] * fftBuf_Myna[0];
    powerSpectrum_Myna[MYNANET_N_FFT / 2] = fftBuf_Myna[1] * fftBuf_Myna[1];
    arm_cmplx_mag_squared_f32(&fftBuf_Myna[2], &powerSpectrum_Myna[1], MYNANET_N_FFT / 2 - 1);

    // OPT-B: Inlined 4-way unrolled filterbank dot-product (64 filters)
    for (int m = 0; m < n_mels_myna; m++) {
      const float *P     = &powerSpectrum_Myna[mynanetMelFiltersFlat[m].startBin];
      const float *W     = &mynanetMelWeightFlat[mynanetMelFiltersFlat[m].offset];
      int          width =  mynanetMelFiltersFlat[m].width;

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
      // librosa power_to_db applies max(1e-10, S)
      float safe_acc = (acc > 1e-10f) ? acc : 1e-10f;
      // IMPORTANT: MynaNet expects input shape (1, 64, 300, 1), i.e., [mels, frames]
      // where the frames dimension varies fastest.
      melOut[m * n_frames_myna + f] = 10.0f * argus_fast_log10f(safe_acc);
    }
  }

  // OPT-C: Exact librosa normalisation (power_to_db + min_max per sample)
  float globalMax;
  uint32_t maxIdx;
  arm_max_f32(melOut, MYNANET_INPUT_SIZE, &globalMax, &maxIdx);

  float top_db = 80.0f;
  float clipMin = globalMax - top_db;

  // 1. Clip to max - 80.0 (simulates librosa power_to_db top_db logic)
  for (int i = 0; i < MYNANET_INPUT_SIZE; i++) {
    if (melOut[i] < clipMin) {
      melOut[i] = clipMin;
    }
  }

  // 2. Compute actual min for per-sample normalisation 
  float globalMin;
  uint32_t minIdx;
  arm_min_f32(melOut, MYNANET_INPUT_SIZE, &globalMin, &minIdx);

  float range = globalMax - globalMin;
  if (range < 1e-8f) range = 1e-8f;
  float invRange = 1.0f / range;

  // 3. Scale to [0, 1]
  for (int i = 0; i < MYNANET_INPUT_SIZE; i++) {
    float v = (melOut[i] - globalMin) * invRange;
    if (v > 1.0f) v = 1.0f;
    if (v < 0.0f) v = 0.0f;
    melOut[i] = v;
  }
}

// ── Audio Capture Pipeline ────────────────────────────────────────────────────
// Busy-wait at exactly 16 kHz using microsecond timestamps.
// 12-bit ADC (0–4095) centred at 2048; sign-extend to ±32767 int16.
void captureAudio() {
  unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
  unsigned long lastSample     = micros();
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (micros() - lastSample < sampleInterval) {}
    lastSample = micros();
    int   val    = analogRead(micPin);
    float sample = ((float)val - 2048.0f) / 2048.0f;
    audioBuffer[i] = (int16_t)(sample * 32767);
  }
}

// ── Class labels (must match training label order) ────────────────────
constexpr int NUM_CLASSES = 12;
const char *SPECIES_NAMES[NUM_CLASSES] = {"Coppersmith Barbet",
                                          "Spotted Dove",
                                          "Collared Kingfisher",
                                          "Asian Koel",
                                          "White-breasted Waterhen",
                                          "Common Iora",
                                          "Large-tailed Nightjar",
                                          "Yellow-vented Bulbul",
                                          "Pied Fantail",
                                          "Olive-backed Sunbird",
                                          "Common Tailorbird",
                                          "White-throated Kingfisher"};

// ── Top-K helper ──────────────────────────────────────────────────────
void topK(TfLiteTensor *out, int n_classes, int k, int *top_idx,
          float *top_prob) {
  for (int i = 0; i < k; i++) {
    top_prob[i] = -1.0f;
    top_idx[i] = -1;
  }
  for (int i = 0; i < n_classes; i++) {
    // Dequantise: float = (int8 - zero_point) * scale
    float p = (out->data.int8[i] - out->params.zero_point) * out->params.scale;
    if (p > top_prob[k - 1]) {
      int j = k - 2;
      while (j >= 0 && p > top_prob[j]) {
        top_prob[j + 1] = top_prob[j];
        top_idx[j + 1] = top_idx[j];
        j--;
      }
      top_prob[j + 1] = p;
      top_idx[j + 1] = i;
    }
  }
}

// ── Main Setup ────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000);

  // ── Built-in LED Setup ────────────────────────────────────────────────
  pinMode(LEDG, OUTPUT);
  digitalWrite(LEDG, HIGH); // Off

  // Enable DWT Cycle Counter for µs-accurate stage timing
  DWT_ENABLE();

  Serial.println(F("\n=== Cascade Argus — Dual-Model Inference Pipeline ==="));
  Serial.println(F("    M7 @ 480 MHz | DWT timing enabled"));

  // ── INA219 initialisation ──────────────────────────────────────────────
  Wire.begin();
  if (ina219.begin()) {
    ina219_ok = true;
    lastPowerMs = millis();
    Serial.println(F("[OK]  INA219 power monitor initialised."));
  } else {
    Serial.println(F("[WARN] INA219 not found — power data unavailable."));
  }

  // ── Initialise FFT & filterbanks ──────────────────────────────────────
  initMelFFT();
  Serial.println(F("[OK] MyDrongo  RFFT-1024 + 16-mel filterbank initialised"));

  initMelFFT_Myna();
  Serial.println(F("[OK] MynaNet RFFT-512  + 64-mel filterbank initialised"));

  // ── Allocate SDRAM ────────────────────────────────────────────────────
  // Layout in SDRAM (sequentially, 16-byte aligned):
  //   [ myna_tensor_arena : 1500 KB ][ myna_features : 75 KB ]
  SDRAM.begin();

  size_t sdramNeeded = (size_t)kMynaArenaSize + (size_t)MYNANET_INPUT_SIZE * sizeof(float) + 32;
  void *raw = SDRAM.malloc(sdramNeeded);
  if (!raw) {
    Serial.println(F("FATAL: SDRAM.malloc failed — insufficient SDRAM."));
    while (1);
  }

  // Align tensor arena to 16 bytes
  myna_tensor_arena = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(raw) + 15) & ~15UL);

  // Place myna_features immediately after the arena (also 16-byte aligned)
  myna_features = reinterpret_cast<float *>(
      (reinterpret_cast<uintptr_t>(myna_tensor_arena) + (size_t)kMynaArenaSize + 15) & ~15UL);

  Serial.print(F("[OK] SDRAM allocated: arena="));
  Serial.print(kMynaArenaSize / 1024);
  Serial.print(F(" KB | myna_features="));
  Serial.print((MYNANET_INPUT_SIZE * sizeof(float)) / 1024);
  Serial.println(F(" KB"));

  // ── Load models ───────────────────────────────────────────────────────
  bad_model = tflite::GetModel(drongonet_micro);
  if (bad_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("FATAL: MyDrongo schema mismatch!"));
    while (1);
  }

  myna_model = tflite::GetModel(MynaNet_1j);
  if (myna_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("FATAL: MynaNet schema mismatch!"));
    while (1);
  }
  Serial.println(F("[OK] Both model flat-buffers loaded from Flash"));

  // ── Operator resolvers ────────────────────────────────────────────────
  static tflite::MicroMutableOpResolver<9> bad_resolver;
  bad_resolver.AddConv2D();
  bad_resolver.AddMaxPool2D();
  bad_resolver.AddMean();
  bad_resolver.AddFullyConnected();
  bad_resolver.AddMul();
  bad_resolver.AddRelu6();
  bad_resolver.AddRelu();   // covers RELU_N1_TO_1 (bc=17, same TFLM kernel)
  bad_resolver.AddTanh();
  bad_resolver.AddSoftmax();

  static tflite::MicroMutableOpResolver<7> myna_resolver;
  myna_resolver.AddConv2D();
  myna_resolver.AddDepthwiseConv2D();
  myna_resolver.AddMaxPool2D();
  myna_resolver.AddMul();
  myna_resolver.AddFullyConnected();
  myna_resolver.AddSoftmax();
  myna_resolver.AddMean();

  Serial.println(F("[OK] TFLM resolvers instantiated"));

  // ── Build interpreters ────────────────────────────────────────────────
  static tflite::MicroInterpreter bad_static_interp(
      bad_model, bad_resolver, bad_tensor_arena, kBadArenaSize);
  bad_interpreter = &bad_static_interp;

  static tflite::MicroInterpreter myna_static_interp(
      myna_model, myna_resolver, myna_tensor_arena, kMynaArenaSize);
  myna_interpreter = &myna_static_interp;

  // ── Allocate tensors ──────────────────────────────────────────────────
  if (bad_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("FATAL: MyDrongo AllocateTensors() failed."));
    while (1);
  }
  if (myna_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("FATAL: MynaNet AllocateTensors() failed."));
    while (1);
  }
  Serial.println(F("[OK] Tensors allocated for both models"));

  bad_input   = bad_interpreter->input(0);
  bad_output  = bad_interpreter->output(0);
  myna_input  = myna_interpreter->input(0);
  myna_output = myna_interpreter->output(0);

  // ── Print arena usage diagnostics ────────────────────────────────────
  Serial.println(F("\n--- Arena Usage ---"));
  Serial.print(F("  MyDrongo  arena used : "));
  Serial.print(bad_interpreter->arena_used_bytes());
  Serial.print(F(" B / "));
  Serial.print(kBadArenaSize);
  Serial.print(F(" B  (slack="));
  Serial.print(kBadArenaSize - (int)bad_interpreter->arena_used_bytes());
  Serial.println(F(" B)"));

  Serial.print(F("  MynaNet arena used: "));
  Serial.print(myna_interpreter->arena_used_bytes());
  Serial.print(F(" B / "));
  Serial.print(kMynaArenaSize);
  Serial.print(F(" B  (slack="));
  Serial.print(kMynaArenaSize - (int)myna_interpreter->arena_used_bytes());
  Serial.println(F(" B)"));

  Serial.println(F("\n[OK] Pipeline ready — entering inference loop.\n"));
}

// ──────────────────────────────────────────────────────────────────────────────
// MAIN LOOP
// ──────────────────────────────────────────────────────────────────────────────
void loop() {
  Serial.println(F("\n╔════════════════════════════════════════╗"));
  Serial.println(F("║          Cascade Argus — Cycle         ║"));
  Serial.println(F("╚════════════════════════════════════════╝"));

  // ── STAGE 1: Audio Capture ────────────────────────────────────────────
  Serial.println(F("[1/4] Capturing 3 s audio @ 16 kHz..."));
  unsigned long captureStart = millis();
  captureAudio();
  unsigned long captureTime  = millis() - captureStart;

  // ── STAGE 2: MyDrongo Mel Spectrogram (16×184) ───────────────────────────
  Serial.println(F("[2/4] Computing MyDrongo mel spectrogram (16-mel × 184 frames)..."));
  DWT_RESET();
  unsigned long melStart    = millis();
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, bad_features);
  uint32_t      melCycles   = DWT_CYCLES();   // hardware cycles — read first
  unsigned long melTime     = millis() - melStart;
  float         melUs       = DWT_US(melCycles);

  // ── STAGE 3: Quantise + fill MyDrongo input tensor ───────────────────────
  DWT_RESET();
  float   bad_scale  = bad_input->params.scale;
  int     bad_zp     = bad_input->params.zero_point;
  for (int i = 0; i < INPUT_SIZE; i++) {
    float   scaled = bad_features[i] / bad_scale + (float)bad_zp;
    if (scaled >  127.0f) scaled =  127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    bad_input->data.int8[i] = (int8_t)scaled;
  }
  uint32_t quantCycles = DWT_CYCLES();
  float    quantUs     = DWT_US(quantCycles);

  // ── STAGE 4: MyDrongo Inference ──────────────────────────────────────────
  Serial.println(F("[3/4] Invoking MyDrongo inference..."));
  DWT_RESET();
  unsigned long inferStart = millis();
  if (bad_interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("[ERR] MyDrongo Invoke() failed."));
    delay(1000);
    return;
  }
  uint32_t      inferCycles = DWT_CYCLES();
  unsigned long inferTime   = millis() - inferStart;
  float         inferUs     = DWT_US(inferCycles);

  // ── Decode MyDrongo output ───────────────────────────────────────────────
  float p_no_bird = 0.0f, p_bird = 0.0f;
  if (bad_output->bytes >= 2) {
    // 2-class softmax: int8[0]=P(no_bird), int8[1]=P(bird)
    p_no_bird = (bad_output->data.int8[0] - bad_output->params.zero_point)
                * bad_output->params.scale;
    p_bird    = (bad_output->data.int8[1] - bad_output->params.zero_point)
                * bad_output->params.scale;
  } else {
    // 1-class sigmoid: int8[0]=P(bird)
    p_bird    = (bad_output->data.int8[0] - bad_output->params.zero_point)
                * bad_output->params.scale;
    p_no_bird = 1.0f - p_bird;
  }

  // Temporal smoothing (3-window ring buffer) — prevents single-spike false positives
  scoreHistory[scoreHistIdx % 3] = p_bird;
  scoreHistIdx++;
  float avgScore    = (scoreHistory[0] + scoreHistory[1] + scoreHistory[2]) / 3.0f;
  bool  birdDetected = (avgScore > 0.6f);

  // Update LED Indicator based on MyDrongo prediction
  if (birdDetected) {
    digitalWrite(LEDG, LOW);  // Turn on green LED
  } else {
    digitalWrite(LEDG, HIGH); // Turn off green LED
  }

  // ── Print per-stage timing log ──────────────────────────────────────────
  Serial.println(F("\n--- MyDrongo Prediction ---"));
  Serial.print(F("Detection  : "));
  Serial.println(birdDetected ? F("BIRD ★") : F("no bird"));
  Serial.print(F("P(bird)    : "));
  Serial.print(p_bird, 4);
  Serial.print(F("  P(no_bird): "));
  Serial.print(p_no_bird, 4);
  Serial.print(F("  AvgScore: "));
  Serial.println(avgScore, 4);

  Serial.print(F("\n--- Timing Log ---"));
  Serial.print(F("\n  capture    = "));
  Serial.print(captureTime);
  Serial.println(F(" ms"));

  Serial.print(F("  mel (BAD)  = "));
  Serial.print(melTime);
  Serial.println(F(" ms"));

  Serial.print(F("  quantise   = "));
  Serial.print(quantUs / 1000.0f, 2);
  Serial.println(F(" ms"));

  Serial.print(F("  infer(BAD) = "));
  Serial.print(inferTime);
  Serial.println(F(" ms"));

  // ── Calculate Latencies for Power Tracking ──────────────────────────────
  unsigned long bad_total_ms   = melTime + (unsigned long)(quantUs / 1000.0f) + inferTime;
  unsigned long myna_total_ms  = 0;
  unsigned long grand_total_ms = bad_total_ms;

  // Variables hoisted so energy block below can access them whether or not
  // MynaNet was triggered.
  unsigned long mynaTime     = 0;
  unsigned long mynaInferTime = 0;

  // STAGE 5 (conditional): MynaNet Mel Spectrogram (64x300)
  if (birdDetected) {
    Serial.println(F("\n[4/4] Bird detected — computing MynaNet mel spectrogram (64 × 300)..."));

    DWT_RESET();
    unsigned long mynaStart  = millis();
    computeMelSpectrogram_Myna(audioBuffer, BUFFER_SIZE, myna_features);
    uint32_t      mynaCycles = DWT_CYCLES();
    mynaTime                 = millis() - mynaStart;   // update hoisted var
    float         mynaUs     = DWT_US(mynaCycles);

    // Spot-check: print the first 8 mel bins of frame 0 and the centre frame
    // to verify numerical plausibility (values should be in [0,1] after normalisation).
    Serial.println(F("\n--- MynaNet Mel Verification ---"));
    Serial.print(F("  Shape: [64 mels × 300 frames]  total="));
    Serial.print(MYNANET_INPUT_SIZE);
    Serial.println(F(" elements"));

    Serial.print(F("  Frame 0  bins[0..7]: "));
    for (int b = 0; b < 8; b++) {
      Serial.print(myna_features[b * MYNANET_TIME_FRAMES + 0], 4);
      if (b < 7) Serial.print(F(", "));
    }
    Serial.println();

    Serial.print(F("  Frame 150 bins[0..7]: "));
    for (int b = 0; b < 8; b++) {
      Serial.print(myna_features[b * MYNANET_TIME_FRAMES + 150], 4);
      if (b < 7) Serial.print(F(", "));
    }
    Serial.println();

    // Verify all values are in [0, 1] after normalisation
    float vMin = myna_features[0], vMax = myna_features[0];
    for (int i = 1; i < MYNANET_INPUT_SIZE; i++) {
      if (myna_features[i] < vMin) vMin = myna_features[i];
      if (myna_features[i] > vMax) vMax = myna_features[i];
    }
    Serial.print(F("  Global min="));
    Serial.print(vMin, 6);
    Serial.print(F("  max="));
    Serial.println(vMax, 6);

    // MynaNet mel timing
    Serial.print(F("\n  mel(Myna) = "));
    Serial.print(mynaTime);
    Serial.println(F(" ms"));

    // ── STAGE 6: Quantise MynaNet Input ──────────────────────────────────
    Serial.println(F("\n[5/6] Quantising MynaNet features..."));
    DWT_RESET();
    float myna_scale = myna_input->params.scale;
    int   myna_zp    = myna_input->params.zero_point;
    for (int i = 0; i < MYNANET_INPUT_SIZE; i++) {
      float s = myna_features[i] / myna_scale + (float)myna_zp;
      if (s >  127.0f) s =  127.0f;
      if (s < -128.0f) s = -128.0f;
      myna_input->data.int8[i] = (int8_t)s;
    }
    uint32_t mynaQuantCycles = DWT_CYCLES();
    float mynaQuantUs = DWT_US(mynaQuantCycles);
    
    Serial.print(F("  quant(Myna)= "));
    Serial.print(mynaQuantUs / 1000.0f, 2);
    Serial.println(F(" ms"));

    // ── STAGE 7: MynaNet Inference ───────────────────────────────────────
    Serial.println(F("\n[6/6] Invoking MynaNet inference..."));
    DWT_RESET();
    unsigned long mynaInferStart = millis();
    if (myna_interpreter->Invoke() != kTfLiteOk) {
      Serial.println(F("[ERR] MynaNet Invoke() failed."));
      delay(1000);
      return;
    }
    uint32_t mynaInferCycles = DWT_CYCLES();
    mynaInferTime            = millis() - mynaInferStart;  // update hoisted var
    float mynaInferUs = DWT_US(mynaInferCycles);

    // ── Decode MynaNet output (Top 3) ────────────────────────────────────
    int top_idx[3];
    float top_prob[3];
    topK(myna_output, NUM_CLASSES, 3, top_idx, top_prob);

    Serial.println(F("\n╔══════════════════════════════╗"));
    Serial.println(F("║   MynaNet — Top 3 Prediction ║"));
    Serial.println(F("╚══════════════════════════════╝"));
    for (int i = 0; i < 3; i++) {
      Serial.print(i + 1);
      Serial.print(F(". "));
      Serial.print((top_idx[i] >= 0 && top_idx[i] < NUM_CLASSES) ? SPECIES_NAMES[top_idx[i]] : "Unknown");
      Serial.print(F("  →  "));
      Serial.print(top_prob[i] * 100.0f, 2);
      Serial.println(F(" %"));
    }

    // ── Timing Breakdown for MynaNet ─────────────────────────────────────
    Serial.println(F("\n─────────── Latency (MynaNet) ────────────"));
    myna_total_ms = mynaTime + (unsigned long)(mynaQuantUs / 1000.0f) + mynaInferTime;
    Serial.print(F("  Mel   : "));
    Serial.print(mynaTime);
    Serial.println(F(" ms"));
    Serial.print(F("  Quant : "));
    Serial.print(mynaQuantUs / 1000.0f, 2);
    Serial.println(F(" ms"));
    Serial.print(F("  Infer : "));
    Serial.print(mynaInferTime);
    Serial.println(F(" ms"));
    Serial.print(F("  Total : "));
    Serial.print(myna_total_ms);
    Serial.println(F(" ms"));

    // ── Cascaded Total Latency ───────────────────────────────────────────
    grand_total_ms = bad_total_ms + myna_total_ms;
    Serial.println(F("─────────── Total Latency ────────────────"));
    Serial.print(F("  BAD   : "));
    Serial.print(bad_total_ms);
    Serial.println(F(" ms"));
    Serial.print(F("  MYNA  : "));
    Serial.print(myna_total_ms);
    Serial.println(F(" ms"));
    Serial.print(F("  Total : "));
    Serial.print(grand_total_ms);
    Serial.println(F(" ms (end-to-end inference after triggered)"));

  } else {
    Serial.println(F("\n[4/6] No bird — MynaNet mel skipped."));
  }

  // POWER & ENERGY BREAKDOWN
  // Cortex-M7 @ 480 MHz busy-waits at 100% duty cycle through mel+infer.
  // A single INA219 snapshot is taken after all compute finishes and used
  // as a representative active-load power figure.
  // E(uJ) = P(mW) * t(ms)   [mW * ms = uW*s = uJ]
  if (ina219_ok) {
    float busVoltage = ina219.getBusVoltage_V();
    float current_mA = ina219.getCurrent_mA();
    float power_mW   = ina219.getPower_mW();

    // Accumulate session energy (whole loop wall-clock time)
    unsigned long now = millis();
    float dt_hours    = (now - lastPowerMs) / 3600000.0f;
    lastPowerMs       = now;
    if (power_mW > 0.0f)
        total_energy_mWh += power_mW * dt_hours;

    // MyDrongo per-stage energies (melTime / inferTime measured above)
    float bad_mel_uJ   = power_mW * (float)melTime;
    float bad_infer_uJ = power_mW * (float)inferTime;
    float bad_total_uJ = power_mW * (float)bad_total_ms;

    // MynaNet per-stage energies (0 when not triggered)
    float myna_mel_uJ   = power_mW * (float)mynaTime;
    float myna_infer_uJ = power_mW * (float)mynaInferTime;
    float myna_total_uJ = power_mW * (float)myna_total_ms;

    float grand_uJ = power_mW * (float)grand_total_ms;

    Serial.println(F("\n─────────── Power & Energy ───────────────"));
    Serial.print(F("  Instantaneous  : "));
    Serial.print(power_mW, 2);
    Serial.print(F(" mW  ("));
    Serial.print(busVoltage, 2);
    Serial.print(F(" V, "));
    Serial.print(current_mA, 2);
    Serial.println(F(" mA)"));

    Serial.println(F("  [Energy - current cycle]"));
    Serial.println(F("    >> MyDrongo (DrongoNet):"));
    Serial.print(F("       Mel computation : "));
    Serial.print(bad_mel_uJ, 2);
    Serial.println(F(" uJ"));
    Serial.print(F("       Inference       : "));
    Serial.print(bad_infer_uJ, 2);
    Serial.println(F(" uJ"));
    Serial.print(F("       Stage total     : "));
    Serial.print(bad_total_uJ, 2);
    Serial.println(F(" uJ"));

    if (birdDetected) {
      Serial.println(F("    >> MynaNet (triggered):"));
      Serial.print(F("       Mel computation : "));
      Serial.print(myna_mel_uJ, 2);
      Serial.println(F(" uJ"));
      Serial.print(F("       Inference       : "));
      Serial.print(myna_infer_uJ, 2);
      Serial.println(F(" uJ"));
      Serial.print(F("       Stage total     : "));
      Serial.print(myna_total_uJ, 2);
      Serial.println(F(" uJ"));
    } else {
      Serial.println(F("    >> MynaNet         : skipped (no bird)"));
    }

    Serial.print(F("    Total cycle energy: "));
    Serial.print(grand_uJ, 2);
    Serial.println(F(" uJ"));
    Serial.print(F("  Session cumulative : "));
    Serial.print(total_energy_mWh, 6);
    Serial.println(F(" mWh"));
    Serial.println(F("─────────────────────────────────────────"));
  }

  Serial.println(F("\n════════════════════════════════════════"));
  delay(2000);
}
