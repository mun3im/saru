#include "Arduino.h"
#include <mbed.h>
#include <mbed_stats.h>

#ifdef abs
#undef abs
#endif

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ==================== Shared ARGUS helpers ====================
// DWT timing macros, fast_log10f, ArgusStageRAM profiling, and the system
// storage breakdown -- see libraries/ARGUS_Common/src/ARGUS_Common.h.
// Install this library (copy libraries/ARGUS_Common into your Arduino
// libraries folder, or open the .ino via the repo checkout) before
// compiling this sketch.
#include <ARGUS_Common.h>

// ==================== Mel Filterbank Tables ====================
#include "mel_tables.h"

// ==================== CMSIS-DSP (Hardware-accelerated FFT) ====
// arm_math.h is provided by the ArduinoCore-mbed BSP for Portenta H7
// (STM32H7, Cortex-M7 with double-precision FPU + SIMD).
// No extra library installation required.
#include <arm_math.h>

// ==================== INA219 Power Monitor ====================
#include <Adafruit_INA219.h>
#include <Wire.h>

Adafruit_INA219 ina219;
float total_energy_mWh = 0.0f;
unsigned long lastPowerMs = 0;

// ==================== Model header ====================
// DrongoNet-micro: 5,312 B flat-buffer, 6 unique op types, 7 graph invocations.
// Folded model: batch-norm is fused into Conv2D kernels; no QUANTIZE/DEQUANTIZE
// boundary nodes (unlike DrongoNet-nano). Direct INT8 input path.
#include "drongonet-micro.h"

// ==================== Model Dimensions ====================
const int MEL_BINS = 16;
const int TIME_FRAMES = 184;
const int INPUT_SIZE = MEL_BINS * TIME_FRAMES;  // 2944 elements

// ==================== Audio Settings ====================
const int micPin = A1;
const int SAMPLE_RATE = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000; // 48 000

// Global audio buffer — lives in BSS (zero-initialised static RAM).
// 48 000 × 2 B = 96 000 B = 93.75 KB.
int16_t audioBuffer[BUFFER_SIZE];

// ==================== TensorFlow Lite ====================
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// -------------------------------------------------------
// Arena sizing rationale:
//   DrongoNet-micro is a folded INT8 BAD model.
//   Batch-norm folded into Conv2D kernels reduces the number
//   of intermediate tensors; arena use is typically ~24–36 KB
//   for 16×184 models — slightly less than nano.
//   Reduce after profiling with the slack figure printed below.
// -------------------------------------------------------
constexpr int kTensorArenaSize = 150 * 1024; // bytes
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // namespace

bool firstLoopDone = false;

// ===========================================================
// CMSIS-DSP FFT instance + buffers
// ===========================================================
static arm_rfft_fast_instance_f32 fftInstance;  // ~76 B state struct

__attribute__((section(".dtcmram")))
static float hannWindow[MEL_N_FFT];    // 1024 floats = 4 KB → DTCM

__attribute__((section(".dtcmram")))
static float fftBuf[MEL_N_FFT];       // 1024 floats = 4 KB → DTCM (in/out)

__attribute__((section(".dtcmram")))
static float powerSpectrum[MEL_N_FFT_BINS]; // 513 floats = 2 KB → DTCM
// Note: fftImag is gone — CMSIS RFFT packs real+imag into fftBuf.

// ===========================================================
// Flat mel filterbank weight table (pre-baked, no runtime div)
// ===========================================================
// Total non-zero bins across all 16 filters = 931
// Memory: 931 × 4 B = 3.6 KB BSS
// Layout: melWeightFlat[melFiltersFlat[m].offset .. +width-1]
//         corresponds to powerSpectrum[startBin .. endBin-1]
#define FLAT_WEIGHT_TOTAL 931

typedef struct {
  int startBin;  // index into powerSpectrum[]
  int width;     // number of weights (endBin - startBin)
  int offset;    // index into melWeightFlat[]
} MelFilterFlat;

static float       melWeightFlat[FLAT_WEIGHT_TOTAL]; // 3.6 KB BSS
static MelFilterFlat melFiltersFlat[MEL_N_MELS];    // 16 × 12 B = 192 B BSS

// ===========================================================
// RMS energy gate + temporal smoothing globals
// ===========================================================
// RMS_THRESHOLD: ADC units (int16 scale ±32767).
// Calibrate from your environment — start at 500, increase if
// fan/rain causes false positives, decrease if bird is quiet.
#define RMS_THRESHOLD   500.0f

// 3-window ring buffer for temporal confidence smoothing.
// A single noisy frame cannot trigger detection.
static float scoreHistory[3] = {0.0f, 0.0f, 0.0f};
static int   scoreHistIdx = 0;

// ===========================================================
// SECTION 1 — TFLM Memory Breakdown
// ===========================================================
// ===========================================================

void printTFLMMemory() {
  Serial.println(F("\n╔══════════════════════════════════════════╗"));
  Serial.println(F("║        TFLM MEMORY BREAKDOWN             ║"));
  Serial.println(F("╚══════════════════════════════════════════╝"));

  // ── 1. Model flat-buffer (Flash) ─────────────────────────
  // drongonet_micro_len is the uint32_t emitted by `xxd -i`.
  // This is the verbatim .tflite byte count stored in Flash.
  Serial.print(F("  [FLASH] Model flat-buffer   : "));
  Serial.print(drongonet_micro_len);
  Serial.print(F(" B  = "));
  Serial.print(drongonet_micro_len / 1024.0f, 2);
  Serial.println(F(" KB  (DrongoNet-micro)"));

  // ── 2. Tensor arena (RAM) ────────────────────────────────
  uint32_t arena_used = (uint32_t)interpreter->arena_used_bytes();
  uint32_t arena_slack = (uint32_t)kTensorArenaSize - arena_used;

  Serial.print(F("  [RAM]   Arena allocated     : "));
  Serial.print(kTensorArenaSize);
  Serial.print(F(" B  = "));
  Serial.print(kTensorArenaSize / 1024.0f, 1);
  Serial.println(F(" KB   (kTensorArenaSize)"));

  Serial.print(F("  [RAM]   Arena used          : "));
  Serial.print(arena_used);
  Serial.print(F(" B  = "));
  Serial.print(arena_used / 1024.0f, 2);
  Serial.println(F(" KB   (HEAD+TAIL after AllocateTensors)"));

  Serial.print(F("  [RAM]   Arena slack (waste) : "));
  Serial.print(arena_slack);
  Serial.print(F(" B  = "));
  Serial.print(arena_slack / 1024.0f, 2);
  Serial.println(F(" KB   ← reduce kTensorArenaSize to (used + 10%% margin)"));

  // ── 3. I/O tensor sizes (sub-region of arena HEAD) ───────
  Serial.print(F("  [RAM]   Input tensor bytes  : "));
  Serial.print(input->bytes);
  Serial.print(F(" B  (INT8, shape=["));
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1)
      Serial.print(',');
  }
  Serial.print(F("]  scale="));
  Serial.print(input->params.scale, 6);
  Serial.print(F("  zp="));
  Serial.println(input->params.zero_point);

  Serial.print(F("  [RAM]   Output tensor bytes : "));
  Serial.print(output->bytes);
  Serial.println(output->bytes >= 2
      ? F(" B  (INT8 softmax [no_bird, bird])")
      : F(" B  (INT8 sigmoid logit)"));

  // ── 4. Runtime overhead estimate ─────────────────────────
  // interpreter + resolver + error_reporter objects are static.
  // MicroMutableOpResolver<6> registers exactly 6 op types for DrongoNet-micro
  //   (folded model: MUL, CONV_2D×2, MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX).
  //   NO QUANTIZE or DEQUANTIZE — batch-norm is folded into Conv2D kernels.
  // This is significantly smaller than AllOpsResolver (~5-15 KB BSS).
  // We cannot measure it precisely at runtime without nm/size
  // tooling, so we note it as a compile-time artefact.
  Serial.println(F(
      "  [FLASH] MicroMutableOpResolver<6> : see .elf map (<<5 KB BSS)"));

  Serial.println(F("  ──────────────────────────────────────────"));
  Serial.print(F("  Minimum viable kTensorArenaSize: "));
  uint32_t recommended = (uint32_t)(arena_used * 1.12f); // +12% margin
  Serial.print(recommended);
  Serial.print(F(" B  = "));
  Serial.print(recommended / 1024.0f, 1);
  Serial.println(F(" KB  (used × 1.12)"));
}

// ===========================================================
// SECTION 2 — Firmware RAM Breakdown
// ===========================================================
// argus_snapshot_ram(), argus_print_system_storage(), and
// argus_print_stage_delta() now live in ARGUS_Common.h -- see the
// #include at the top of this file.

void printFirmwareRAM() {
  Serial.println(F("\n╔══════════════════════════════════════════╗"));
  Serial.println(F("║      FIRMWARE RAM BREAKDOWN              ║"));
  Serial.println(F("╚══════════════════════════════════════════╝"));

  // ── Static / BSS (compile-time known) ────────────────────
  Serial.println(F("  [BSS / .data  — static globals]"));
  Serial.print(F("    audioBuffer[48000×int16]  : "));
  Serial.print((uint32_t)BUFFER_SIZE * sizeof(int16_t));
  Serial.println(F(" B  = 96000 B (93.75 KB)"));

  Serial.print(F("    tensor_arena[kArenaSize]   : "));
  Serial.print((uint32_t)kTensorArenaSize);
  Serial.print(F(" B  = "));
  Serial.print(kTensorArenaSize / 1024.0f, 0);
  Serial.println(F(" KB"));

  Serial.println(F("    MicroInterpreter (static)  : ~1200 B  (see .map)"));
  Serial.println(F("    MicroMutableOpResolver<6>  : <<5 KB  (6 types: MUL,CONV_2D,MAXPOOL,MEAN,FC,SOFTMAX — folded, no QUANT/DEQUANT)"));

  // ── Heap (dynamic, Mbed-tracked) ─────────────────────────
  mbed_stats_heap_t heap;
  mbed_stats_heap_get(&heap);
  Serial.println(F("  [HEAP  — dynamic Mbed allocations]"));
  Serial.print(F("    current live bytes         : "));
  Serial.print(heap.current_size);
  Serial.println(F(" B"));
  Serial.print(F("    peak (max ever)            : "));
  Serial.print(heap.max_size);
  Serial.print(F(" B  = "));
  Serial.print(heap.max_size / 1024.0f, 2);
  Serial.println(F(" KB"));
  Serial.print(F("    current alloc count        : "));
  Serial.println(heap.alloc_cnt); // field: alloc_cnt (not total_alloc_cnt)
  Serial.print(F("    alloc failures             : "));
  Serial.println(heap.alloc_fail_cnt); // non-zero = out-of-memory events!

  // ── Stack (Mbed HWM per thread) ──────────────────────────
  Serial.println(F("  [STACK — Mbed OS high-water marks per thread]"));
  int cnt = osThreadGetCount();
  if (cnt > 0 && cnt <= 32) {
    mbed_stats_stack_t *stk =
        (mbed_stats_stack_t *)malloc(cnt * sizeof(mbed_stats_stack_t));
    if (stk) {
      int got = mbed_stats_stack_get_each(stk, cnt);
      for (int i = 0; i < got; i++) {
        uint32_t used = stk[i].max_size;
        uint32_t res = stk[i].reserved_size;
        uint32_t free_bytes = (res > used) ? (res - used) : 0;
        Serial.print(F("    Thread 0x"));
        Serial.print(stk[i].thread_id, HEX);
        Serial.print(F("  HWM="));
        Serial.print(used);
        Serial.print(F("B / reserved="));
        Serial.print(res);
        Serial.print(F("B  free="));
        Serial.print(free_bytes);
        Serial.println(F("B"));
      }
      free(stk);
    }
  }
}

// ===========================================================
// CMSIS-DSP FFT + Hann Window Initialisation
// ===========================================================
void initMelFFT() {
  // ── Hann window ────────────────────────────────────────────
  for (int i = 0; i < MEL_N_FFT; i++) {
    hannWindow[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)MEL_N_FFT));
  }

  // ── CMSIS Real-FFT instance ────────────────────────────────
  arm_rfft_fast_init_f32(&fftInstance, MEL_N_FFT);

  // ── Pre-bake flat filterbank weight table ──────────────────
  int offset = 0;
  for (int m = 0; m < MEL_N_MELS; m++) {
    int s = melFilters[m].startBin;
    int p = melFilters[m].peakBin;
    int e = melFilters[m].endBin;
    int width = e - s;  // number of non-zero bins
    melFiltersFlat[m].startBin = s;
    melFiltersFlat[m].width    = width;
    melFiltersFlat[m].offset   = offset;
    float rise_inv  = (p > s) ? 1.0f / (float)(p - s) : 0.0f;
    float fall_inv  = (e > p) ? 1.0f / (float)(e - p) : 0.0f;
    for (int k = s; k < e; k++) {
      float w;
      if (k < p)       w = (float)(k - s) * rise_inv;  // rising
      else if (k == p) w = 1.0f;                        // peak
      else             w = (float)(e - k) * fall_inv;  // falling
      melWeightFlat[offset + (k - s)] = w;
    }
    offset += width;
  }
}

// ===========================================================
// Mel Spectrogram — FFT-based, matching training pipeline
// ===========================================================
void computeMelSpectrogram(int16_t *audio, int audioLength, float *melOut) {

  for (int f = 0; f < TIME_FRAMES; f++) {
    int frameStart = f * MEL_HOP_LENGTH;
    int validSamples = audioLength - frameStart;
    if (validSamples > MEL_N_FFT) validSamples = MEL_N_FFT;

    // ── OPT-A: Fused int16→float + Hann window (single pass) ──────
    const float INV_32768 = 1.0f / 32768.0f;
    const int16_t *src = &audio[frameStart];
    for (int i = 0; i < validSamples; i++) {
      fftBuf[i] = (float)src[i] * (hannWindow[i] * INV_32768);
    }
    // Zero-pad the tail (only the last frame is ever short).
    if (validSamples < MEL_N_FFT) {
      memset(&fftBuf[validSamples], 0,
             (MEL_N_FFT - validSamples) * sizeof(float));
    }

    // ── 3. FFT (CMSIS hardware-accelerated) ──────────────────
    arm_rfft_fast_f32(&fftInstance, fftBuf, fftBuf, 0 /*forward*/);

    // ── 4. Power spectrum |X[k]|² for k = 0..N/2 ─────────────────
    // DC bin (k=0): fftBuf[0]^2
    powerSpectrum[0] = fftBuf[0] * fftBuf[0];
    // Nyquist bin (k=N/2): fftBuf[1]^2
    powerSpectrum[MEL_N_FFT / 2] = fftBuf[1] * fftBuf[1];
    // Bins 1..N/2-1: arm_cmplx_mag_squared_f32 computes Re²+Im²
    // for each complex pair, 2 samples/cycle via Cortex-M7 SIMD.
    arm_cmplx_mag_squared_f32(&fftBuf[2], &powerSpectrum[1],
                              MEL_N_FFT / 2 - 1);

    // ── OPT-B: Inlined filterbank — 4-way unrolled dot-product ────
    float *melRow = &melOut[f * MEL_N_MELS];
    for (int m = 0; m < MEL_N_MELS; m++) {
      const float *P = &powerSpectrum[melFiltersFlat[m].startBin];
      const float *W = &melWeightFlat[melFiltersFlat[m].offset];
      int   width    =  melFiltersFlat[m].width;

      float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
      int k = 0;
      // 4-wide unrolled body
      for (; k <= width - 4; k += 4) {
        acc0 += P[k+0] * W[k+0];
        acc1 += P[k+1] * W[k+1];
        acc2 += P[k+2] * W[k+2];
        acc3 += P[k+3] * W[k+3];
      }
      // Scalar tail for remaining 0-3 bins
      float acc = (acc0 + acc1) + (acc2 + acc3);
      for (; k < width; k++) {
        acc += P[k] * W[k];
      }

      // ── 6. Power → dB (fast IEEE-754 approximation) ─────
      melRow[m] = 10.0f * argus_fast_log10f(acc + 1e-10f);
    }
  } // end frame loop

  // ── 7. Find min/max via SIMD ──────────────────────────────────
  float globalMin, globalMax;
  uint32_t minIdx, maxIdx;
  arm_min_f32(melOut, INPUT_SIZE, &globalMin, &minIdx);
  arm_max_f32(melOut, INPUT_SIZE, &globalMax, &maxIdx);

  // ── OPT-C: SIMD normalisation ─────────────────────────────────
  // Per-sample min-max → [0, 1] matching DrongoNet training pipeline.
  // arm_offset_f32: out[i] = in[i] + offset    (SIMD, 4 float/cycle)
  // arm_scale_f32:  out[i] = in[i] * scale     (SIMD, 4 float/cycle)
  // Replaces the scalar loop + two per-element clamp branches.
  // Note: clamp to [0,1] is still needed for numeric safety.
  float range = globalMax - globalMin;
  if (range < 1e-6f) range = 1e-6f;
  float invRange = 1.0f / range;

  // Subtract globalMin from all elements (SIMD)
  arm_offset_f32(melOut, -globalMin, melOut, INPUT_SIZE);
  // Multiply all elements by 1/range (SIMD)
  arm_scale_f32(melOut, invRange, melOut, INPUT_SIZE);
  // Clamp to [0, 1] — arm_clip_f32 if available, else scalar
  // (this is only 2944 compares; cost is ~0.012 ms, acceptable)
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
// NOTE: Portenta H7 default ADC is 12-bit (0–4095).
// If you called analogReadResolution(10), change 2048→512.
void captureAudio() {
  unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
  unsigned long lastSample = micros();
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (micros() - lastSample < sampleInterval) {
    }
    lastSample = micros();
    int val = analogRead(micPin);
    // 12-bit ADC: center at 2048, full-scale ±2048
    float sample = ((float)val - 2048.0f) / 2048.0f;
    audioBuffer[i] = (int16_t)(sample * 32767);
  }
}

// ===========================================================
// SETUP
// ===========================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  delay(2000);

  // ── Enable DWT Cycle Counter ──────────────────────────────
  DWT_ENABLE();

  Serial.println(F("=== MyDrongo [DrongoNet-micro] — Memory Profiling Build ==="));

  // INA219
  Wire.begin();
  if (!ina219.begin()) {
    Serial.println(F("[WARN] INA219 not found — power data unavailable."));
  } else {
    Serial.println(F("[OK]  INA219 initialised."));
    lastPowerMs = millis();
  }

  // Load model
  model = tflite::GetModel(drongonet_micro);
  if (!model) {
    Serial.println(F("[ERR] Model load failed."));
    while (1) {
    }
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("[ERR] Schema mismatch: "));
    Serial.print(model->version());
    Serial.print(F(" vs "));
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) {
    }
  }

  // DrongoNet-micro EXACT op set — decoded from flatbuffer operator_codes table
  // (python flatbuffer parse, 6 OperatorCode entries, 7 operator invocations):
  //
  //   idx  opcode  name             version   invocations
  //   ---  ------  ---------------  -------   -----------
  //    0     18    MUL              v2        freq_emph_folded/mul        (×1)
  //    1      3    CONV_2D          v3        conv2d, conv2d_1            (×2)
  //    2     17    MAX_POOL_2D      v2        max_pooling2d               (×1)
  //    3     40    MEAN             v2        global_average_pooling2d    (×1)
  //    4      9    FULLY_CONNECTED  v4        dense head                  (×1)
  //    5     25    SOFTMAX          v2        output                      (×1)
  //
  // NOTE: Folded model — NO QUANTIZE (114) or DEQUANTIZE (6) boundary nodes.
  //       Batch-norm is folded into Conv2D kernels during MLIR export.
  //       Input tensor is directly INT8; model starts with MUL on INT8 data.
  //       Total: 6 unique op types, 7 graph invocations.
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddMul();             // opcode  18 v2
  resolver.AddConv2D();          // opcode   3 v3  (used twice in graph)
  resolver.AddMaxPool2D();       // opcode  17 v2
  resolver.AddMean();            // opcode  40 v2
  resolver.AddFullyConnected();  // opcode   9 v4
  resolver.AddSoftmax();         // opcode  25 v2

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] AllocateTensors() failed."));
    Serial.print(F("      Arena size = "));
    Serial.println(kTensorArenaSize);
    while (1) {
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // ── Print memory sections now ────────────────────────────
  // TFLM breakdown is valid immediately after AllocateTensors.
  printTFLMMemory();

  // System Hardware Storage Breakdown (Flash & RAM limits).
  argus_print_system_storage();

  // Firmware RAM baseline (before any loop execution).
  printFirmwareRAM();

  // ── Initialise CMSIS-DSP FFT + Hann window ───────────────
  initMelFFT();
  Serial.println(F("[OK]  CMSIS-DSP RFFT + Hann window initialised."));
  Serial.print(F("      FFT size: ")); Serial.print(MEL_N_FFT);
  Serial.print(F("  hop: ")); Serial.print(MEL_HOP_LENGTH);
  Serial.print(F("  mels: ")); Serial.println(MEL_N_MELS);

  Serial.println(F("\n[OK]  DRONGO inference engine ready."));
}

// ===========================================================
// MAIN LOOP
// ===========================================================
void loop() {
  // ── One-time startup info ─────────────────────────────────
  if (!firstLoopDone) {
    Serial.println();
    Serial.println(F("========== PIPELINE BUFFER INFO =========="));
    Serial.print(F("audioBuffer        : "));
    Serial.print(BUFFER_SIZE * sizeof(int16_t));
    Serial.println(F(" B  (global BSS)"));
    Serial.print(F("melFeatures[2944]  : "));
    Serial.print(INPUT_SIZE * sizeof(float));
    Serial.println(F(" B  (stack, inside loop)"));
    Serial.print(F("tensor input buf   : "));
    Serial.print(input->bytes);
    Serial.println(F(" B  (arena HEAD)"));
    Serial.print(F("tensor output buf  : "));
    Serial.print(output->bytes);
    Serial.println(F(" B  (arena HEAD)"));
    Serial.print(F("Input tensor dims  : ["));
    for (int i = 0; i < input->dims->size; i++) {
      Serial.print(input->dims->data[i]);
      if (i < input->dims->size - 1)
        Serial.print(',');
    }
    Serial.println(F("]"));
    Serial.println(F("==========================================\n"));
    firstLoopDone = true;
  }

  unsigned long startTime = millis();

  // ── STAGE 1: Audio Capture ───────────────────────────────
  ArgusStageRAM before_capture = argus_snapshot_ram();
  captureAudio();
  ArgusStageRAM after_capture = argus_snapshot_ram();
  unsigned long captureTime = millis() - startTime;

  // ── RMS Energy Gate ───────────────────────────────────────
  // Compute RMS of the captured int16 audio buffer.
  // If below RMS_THRESHOLD, the window is silence/background
  // — skip mel computation and inference entirely.
  // This prevents any signal from being amplified by per-sample
  // min-max normalisation and fed into the model as a false positive.
  float rmsAccum = 0.0f;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float s = (float)audioBuffer[i];
    rmsAccum += s * s;
  }
  float rms = sqrtf(rmsAccum / (float)BUFFER_SIZE);

  if (rms < RMS_THRESHOLD) {
    Serial.print(F("[GATE] Silent window — RMS="));
    Serial.print(rms, 1);
    Serial.print(F(" < threshold="));
    Serial.print(RMS_THRESHOLD, 0);
    Serial.println(F(" — inference skipped."));
    lastPowerMs = millis();   // ← advance timestamp; prevents idle period
                              //   being absorbed into next active frame's dt
    delay(100);
    return;
  }

  // ── STAGE 2: Mel Spectrogram (FFT-based) ─────────────────
  static float melFeatures[INPUT_SIZE];

  ArgusStageRAM before_mel = argus_snapshot_ram();

  // DWT-based cycle measurement for µs-accurate mel timing.
  // millis() has 1 ms resolution and can be inflated by OS scheduler
  // ticks and UART FIFO flushing.  DWT gives single-cycle accuracy.
  DWT_RESET();
  unsigned long melStart = millis();   // kept for legacy display
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, melFeatures);
  uint32_t melCycles = DWT_CYCLES();  // read immediately after return
  unsigned long melTime = millis() - melStart;
  float melUs = DWT_US(melCycles);    // true µs from hardware counter

  ArgusStageRAM after_mel = argus_snapshot_ram();

  // ── STAGE 3: Quantise → Fill tensor ──────────────────────
  for (int i = 0; i < INPUT_SIZE; i++) {
    float scaled =
        melFeatures[i] / input->params.scale + input->params.zero_point;
    if (scaled > 127.0f)
      scaled = 127.0f;
    if (scaled < -128.0f)
      scaled = -128.0f;
    input->data.int8[i] = (int8_t)scaled;
  }

  // ── STAGE 4: Inference ───────────────────────────────────
  ArgusStageRAM before_infer = argus_snapshot_ram();
  DWT_RESET();
  unsigned long inferStart = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  uint32_t inferCycles = DWT_CYCLES();   // hardware-accurate cycle count
  unsigned long inferTime = millis() - inferStart;
  float inferUs = DWT_US(inferCycles);   // true µs independent of OS jitter
  ArgusStageRAM after_infer = argus_snapshot_ram();

  if (invoke_status != kTfLiteOk) {
    Serial.println(F("[ERR] Invoke() failed."));
    return;
  }

  // ── STAGE 5: Decode output ───────────────────────────────
  // Handle both 2-class softmax (DrongoNet-nano: Dense(2,softmax))
  // and 1-class sigmoid (BAD_tiny: Dense(1,sigmoid)).
  float result = 0.0f;

  if (output->bytes >= 2) {
    // 2-class softmax: output[0]=P(no_bird), output[1]=P(bird)
    float prob_no_bird = (output->data.int8[0] - output->params.zero_point)
                         * output->params.scale;
    float prob_bird    = (output->data.int8[1] - output->params.zero_point)
                         * output->params.scale;
    result = prob_bird;

    // One-time diagnostic
    static bool outputDiagPrinted = false;
    if (!outputDiagPrinted) {
      outputDiagPrinted = true;
      Serial.println(F("\n--- OUTPUT DIAGNOSTIC ---"));
      Serial.print(F("  output->bytes = "));
      Serial.println(output->bytes);
      Serial.print(F("  scale = "));
      Serial.print(output->params.scale, 6);
      Serial.print(F("  zp = "));
      Serial.println(output->params.zero_point);
      Serial.print(F("  raw int8[0] = "));
      Serial.print(output->data.int8[0]);
      Serial.print(F("  raw int8[1] = "));
      Serial.println(output->data.int8[1]);
      Serial.print(F("  P(no_bird) = "));
      Serial.print(prob_no_bird, 4);
      Serial.print(F("  P(bird) = "));
      Serial.println(prob_bird, 4);
      Serial.println(F("--- END OUTPUT DIAGNOSTIC ---\n"));
    }
  } else {
    // 1-class sigmoid: output[0] = P(bird)
    result = (output->data.int8[0] - output->params.zero_point)
             * output->params.scale;
  }

  // ── Temporal smoothing (3-window ring buffer) ─────────────
  // Averages the last 3 raw scores (~9 seconds of audio).
  // A single transient noise spike cannot alone trigger a BIRD
  // detection. Threshold raised to 0.65 for extra margin.
  scoreHistory[scoreHistIdx % 3] = result;
  scoreHistIdx++;
  float avgScore = (scoreHistory[0] + scoreHistory[1] + scoreHistory[2]) / 3.0f;
  bool birdDetected = (avgScore > 0.6f);
  digitalWrite(LED_BUILTIN, birdDetected ? HIGH : LOW);

  // ── Print inference result ────────────────────────────────
  Serial.print(F("Detection: "));
  Serial.print(birdDetected ? F("BIRD") : F("no bird"));
  Serial.print(F(" | RawScore: "));
  Serial.print(result, 4);
  Serial.print(F(" | AvgScore: "));
  Serial.print(avgScore, 4);
  Serial.print(F(" | RMS: "));
  Serial.print(rms, 0);
  Serial.print(F(" | capture="));
  Serial.print(captureTime);
  Serial.print(F("ms mel="));
  Serial.print(melTime);
  Serial.print(F("ms infer="));
  Serial.print(inferTime);
  // DWT timing for both mel and inference — hardware cycle counter,
  // independent of millis() / OS scheduler jitter.
  Serial.print(F("ms | mel_cyc="));
  Serial.print(melCycles);
  Serial.print(F(" mel_us="));
  Serial.print(melUs, 1);
  Serial.print(F(" | infer_cyc="));
  Serial.print(inferCycles);
  Serial.print(F(" infer_us="));
  Serial.print(inferUs, 1);
  Serial.println(F(" µs"));

  // ── Per-stage transient RAM deltas ───────────────────────
  // Print once every N loops to avoid flooding Serial.
  // Change LOG_INTERVAL to suit.
  static uint32_t loopCount = 0;
  const uint32_t LOG_INTERVAL = 10; // print every 10 inferences
  if (++loopCount % LOG_INTERVAL == 0) {
    Serial.println(F("\n--- Transient RAM per pipeline stage ---"));
    argus_print_stage_delta("1_capture", before_capture, after_capture);
    argus_print_stage_delta("2_mel    ", before_mel, after_mel);
    argus_print_stage_delta("3_infer  ", before_infer, after_infer);

    // Refresh heap summary
    mbed_stats_heap_t heap;
    mbed_stats_heap_get(&heap);
    Serial.print(F("  Heap: current="));
    Serial.print(heap.current_size);
    Serial.print(F("B  peak="));
    Serial.print(heap.max_size);
    Serial.print(F("B  fail="));
    Serial.println(heap.alloc_fail_cnt);
    Serial.println(F("----------------------------------------\n"));
  }

  // ── Power & Energy Breakdown ─────────────────────────────
  // Single INA219 snapshot taken after all compute stages are done.
  // The Cortex-M7 runs at 480 MHz duty cycle throughout mel+infer,
  // so instantaneous power is a good proxy for the active load.
  // E(µJ) = P(mW) × t(ms)   [mW × ms = µW·s = µJ]
  float busVoltage = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW   = ina219.getPower_mW();

  // Accumulate session energy (whole loop wall time including capture)
  unsigned long now = millis();
  float dt_hours = (now - lastPowerMs) / 3600000.0f;
  lastPowerMs = now;
  if (power_mW > 0.0f)
    total_energy_mWh += power_mW * dt_hours;

  // Per-stage energy: use DWT-measured µs for accuracy.
  // E(µJ) = P(mW) × t(ms) = P(mW) × t(µs)/1000
  // Using DWT µs avoids the 1 ms millis() granularity error on short infer times.
  float energy_mel_uJ   = power_mW * (melUs   / 1000.0f); // mel computation
  float energy_infer_uJ = power_mW * (inferUs / 1000.0f); // inference
  float energy_total_uJ = energy_mel_uJ + energy_infer_uJ; // active compute

  Serial.println(F("\n─────────── Power & Energy ──────────────"));
  Serial.print(F("  Instantaneous  : "));
  Serial.print(power_mW, 2);
  Serial.print(F(" mW  ("));
  Serial.print(busVoltage, 2);
  Serial.print(F(" V, "));
  Serial.print(current_mA, 2);
  Serial.println(F(" mA)"));
  Serial.println(F("  [Energy — current cycle, DWT-timed]"));
  Serial.print(F("    Mel computation : "));
  Serial.print(energy_mel_uJ, 2);
  Serial.println(F(" µJ"));
  Serial.print(F("    Inference       : "));
  Serial.print(energy_infer_uJ, 2);
  Serial.println(F(" µJ"));
  Serial.print(F("    Total (active)  : "));
  Serial.print(energy_total_uJ, 2);
  Serial.println(F(" µJ"));
  Serial.print(F("  Session cumulative: "));
  Serial.print(total_energy_mWh, 6);
  Serial.println(F(" mWh"));
  Serial.println(F("─────────────────────────────────────────"));
  Serial.println();

  delay(100);
}
