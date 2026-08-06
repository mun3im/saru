// h7_mynanet_m7_instrumented.ino — MynaNet 1o characterization on M7
//
// Tier-2 species classifier (12 Malaysian garden birds) run standalone on
// the Cortex-M7 with live mic input, DWT-cycle-accurate per-stage latency,
// and optional INA219 energy accounting -- the MynaNet counterpart of
// h7_drongonet_m7_instrumented.
//
// Measures, per 3 s window: mel-spectrogram-only, inference-only, and
// combined latency, plus energy for each stage when the INA219 is wired.
//
// Model: MynaNet 1o (MBV3 + MatchboxNet FCN epilogue), INT8, 64x300 mel
// input -> 12-class softmax. Now the adopted MynaNet: ~same accuracy as
// the older 1j baseline at 38% fewer params (120,500 vs 193,396) and
// 193 KB vs 267 KB. See mynanet_1o.h for exactly which trained artifact
// this is.
//
// Differences from the DrongoNet sketches that matter here:
//   - Arena lives in SDRAM (1.5 MB). MynaNet's first blocks run at full
//     64x300x32 resolution -- a single activation tensor is ~600 KB, so
//     this cannot fit M7's ~523 KB internal SRAM. Same approach (and
//     same arena size) as the existing src/MynaNet/MynaNet.ino.
//   - SCB_CleanInvalidateDCache() after writing the input tensor. The M7
//     writes the tensor through its D-cache, but the kernels read the
//     SDRAM-backed arena; without the flush they can see stale data. This
//     is lifted from src/MynaNet/MynaNet.ino, which documents the same
//     hazard.
//   - Mel front-end differs from DrongoNet's: n_fft=512 (not 1024),
//     hop=160, win_length=400 hann centred in the 512-pt frame, 64 mels,
//     fmin=0, fmax=8000, librosa slaney-normalised weights stored
//     verbatim (see mynanet_mel_tables.h), per-clip power_to_db(ref=max)
//     then clip/normalise against FIXED training dB bounds.

#include <arm_math.h>  // CMSIS-DSP, bundled with the Portenta board package

#ifdef abs
#undef abs
#endif

#include "Chirale_TensorFlowLite.h"  // tested swapping this for the old
                                     // Arduino_TensorFlowLite (2.4.0-ALPHA)
                                     // on 2026-08-03 to see if the Invoke()
                                     // hang on SDRAM arenas was
                                     // Chirale-specific -- it wasn't, hung
                                     // identically either way (see
                                     // argus_mynanet_1o_sdram_hang memory).
                                     // Kept on Chirale since it's the
                                     // better-maintained library and this
                                     // test ruled out the library as the
                                     // cause anyway.
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <SDRAM.h>

#include "mynanet_mel_tables.h"
#include "mynanet_1o.h"

// DWT cycle-counter macros -- see h7_drongonet_m7_instrumented for why
// these are inlined rather than pulled from ARGUS_Common.h (that header's
// argus_print_system_storage() declares __etext/__data_start__/__data_end__
// in a way that conflicts with this snap's mbed_portenta 3.3.0 core).
#define DWT_ENABLE()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; \
                           DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while(0)
#define DWT_RESET()   do { DWT->CYCCNT = 0; } while(0)
#define DWT_CYCLES()  (DWT->CYCCNT)
#define DWT_US(cyc)   ((float)(cyc) / 480.0f)   // us at 480 MHz

#include <Adafruit_INA219.h>
#include <Wire.h>

Adafruit_INA219 ina219;
bool ina219_ok = false;

// ==================== SDRAM cacheability ====================
// Stock behaviour (0): SDRAM at 0x60000000 is left NON-CACHEABLE on M7.
// Verified in the core sources, not assumed -- mbed_portenta 3.3.0's
// SDRAM.cpp has its MPU-config blocks commented out behind `#if 0`, and
// even 4.6.0 only calls MPU_Config() `#ifdef CORE_CM4` and sets
// MPU_ACCESS_NOT_CACHEABLE there. So every arena access during Invoke()
// goes to external SDRAM uncached, which is dramatically slower than
// internal SRAM for a CNN's access pattern.
//
// Setting this to 1 marks the 8 MB SDRAM region write-back cacheable +
// bufferable (TEX=1, C=1, B=1) -- what 3.3.0's disabled `#if 0` block was
// meant to do. Safe here because only the CPU touches the arena (no DMA
// or second-core access to it), and the sketch already cleans the D-cache
// before Invoke(). Kept OFF by default so the baseline numbers describe
// stock behaviour; flip to 1 to measure the cached case.
#define ENABLE_SDRAM_CACHE 1

#if ENABLE_SDRAM_CACHE
static void configureSdramCacheable() {
  MPU_Region_InitTypeDef mpu = {0};
  HAL_MPU_Disable();
  mpu.Enable           = MPU_REGION_ENABLE;
  mpu.BaseAddress      = 0x60000000;
  mpu.Size             = MPU_REGION_SIZE_8MB;
  mpu.Number           = MPU_REGION_NUMBER5;
  mpu.TypeExtField     = MPU_TEX_LEVEL1;          // + C=1,B=1 -> write-back, write-allocate
  mpu.AccessPermission = MPU_REGION_FULL_ACCESS;
  mpu.DisableExec      = MPU_INSTRUCTION_ACCESS_DISABLE;
  mpu.IsShareable      = MPU_ACCESS_NOT_SHAREABLE;
  mpu.IsCacheable      = MPU_ACCESS_CACHEABLE;
  mpu.IsBufferable     = MPU_ACCESS_BUFFERABLE;
  mpu.SubRegionDisable = 0x00;
  HAL_MPU_ConfigRegion(&mpu);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}
#endif

// ==================== Fault handler (diagnose the SDRAM Invoke() hang) ====
// This sketch's Invoke() has a documented history (see the
// argus_mynanet_1o_sdram_hang memory) of silently hanging with an
// SDRAM-backed arena, and the board also goes unresponsive to
// touch-reset shortly after -- a pattern more consistent with a
// HardFault/BusFault (BusFault being the prime suspect given SDRAM is
// the thing that changed) than a true infinite loop. No debug probe is
// available, so this catches the fault in software instead.
//
// First attempt was overriding HardFault_Handler/BusFault_Handler/
// UsageFault_Handler/MemManage_Handler directly -- doesn't link, mbed's
// own except.S already provides STRONG (non-weak) definitions of all
// four, routing into mbed_fault_handler(). That handler itself is also
// a strong symbol, but mbed_error_hook() -- which mbed_fault_handler()
// calls into internally with a summarized mbed_error_ctx -- IS weak and
// documented as the supported override point, so that's the hook used
// here instead.
//
// The remaining problem: mbed_error_hook() still runs at fault-handler
// priority (higher than any configurable interrupt), which blocks the
// USB CDC interrupt from ever firing -- so Serial output attempted
// directly from inside it would just sit in a TX buffer forever,
// indistinguishable from the original silent hang, and mbed's own halt
// loop after the hook returns (MBED_CONF_PLATFORM_FATAL_ERROR_AUTO_
// REBOOT_ENABLED is 0 in this core build, confirmed in mbed_config.h --
// it halts, doesn't reboot) never lowers back out of that priority
// either. So: don't try to print from here at all. Read the fault
// registers directly (plain memory reads, no interrupt/priority
// concern) and stash them in the STM32H7's RTC backup registers
// (RTC->BKPxR, 32 available) -- these survive any warm reset, including
// the double-tap physical reset already used to recover from this exact
// hang every time. setup() checks for and prints them on the NEXT boot,
// using Serial once it's back in completely normal boot context. The
// existing double-tap recovery cycle becomes the delivery mechanism.
//
// (mbed's own MBED_CONF_PLATFORM_CRASH_CAPTURE_ENABLED feature would
// have done this automatically, but it's 0 in this exact core/variant
// build -- confirmed in mbed_config.h -- and that's baked into the
// precompiled libmbed.a, not something a sketch-level #define can turn
// on.)
extern "C" void mbed_error_hook(const mbed_error_ctx *error_context) {
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_RTC_ENABLE();
  RTC->BKP1R = (uint32_t)error_context->error_status;
  RTC->BKP2R = error_context->error_address;
  RTC->BKP3R = error_context->error_value;
  RTC->BKP4R = SCB->CFSR;
  RTC->BKP5R = SCB->HFSR;
  RTC->BKP6R = SCB->BFAR;
  RTC->BKP7R = SCB->MMFAR;
  // Magic marker written LAST so an incomplete write (e.g. a second,
  // worse fault happening mid-hook) can't look like valid complete data.
  RTC->BKP0R = 0xFA17DEADUL;
}

// Called from setup(), before anything else -- prints and clears any
// fault recorded by mbed_error_hook() above during the previous run.
void checkAndReportPreviousFault() {
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_RTC_ENABLE();
  if (RTC->BKP0R != 0xFA17DEADUL) {
    Serial.println(F("[OK]  No fault recorded from previous run."));
    return;
  }

  uint32_t errStatus  = RTC->BKP1R;
  uint32_t errAddress = RTC->BKP2R;
  uint32_t errValue   = RTC->BKP3R;
  uint32_t cfsr       = RTC->BKP4R;
  uint32_t hfsr       = RTC->BKP5R;
  uint32_t bfar       = RTC->BKP6R;
  uint32_t mmfar      = RTC->BKP7R;
  RTC->BKP0R = 0;  // consume it -- don't reprint on the next normal boot

  Serial.println();
  Serial.println(F("*** FAULT RECORDED FROM PREVIOUS RUN ***"));
  Serial.print(F("mbed error_status  = 0x")); Serial.println(errStatus, HEX);
  Serial.print(F("mbed error_address = 0x")); Serial.println(errAddress, HEX);
  Serial.print(F("mbed error_value   = 0x")); Serial.println(errValue, HEX);
  Serial.print(F("CFSR = 0x")); Serial.println(cfsr, HEX);
  Serial.print(F("HFSR = 0x")); Serial.println(hfsr, HEX);

  // BFARVALID = CFSR bit 15, MMARVALID = CFSR bit 7 -- BFAR/MMFAR only
  // hold meaningful data when their respective VALID bit is set.
  if (cfsr & (1UL << 15)) {
    Serial.print(F("BFAR = 0x")); Serial.print(bfar, HEX);
    Serial.println(F("  <- faulting address (valid)"));
  } else {
    Serial.println(F("BFAR invalid (BFARVALID clear)"));
  }
  if (cfsr & (1UL << 7)) {
    Serial.print(F("MMFAR= 0x")); Serial.print(mmfar, HEX);
    Serial.println(F("  <- faulting address (valid)"));
  } else {
    Serial.println(F("MMFAR invalid (MMARVALID clear)"));
  }

  uint8_t  mmfsr = cfsr & 0xFF;
  uint8_t  bfsr  = (cfsr >> 8) & 0xFF;
  uint16_t ufsr  = (cfsr >> 16) & 0xFFFF;
  Serial.print(F("MMFSR=0x")); Serial.print(mmfsr, HEX);
  Serial.print(F(" BFSR=0x"));  Serial.print(bfsr, HEX);
  Serial.print(F(" UFSR=0x"));  Serial.println(ufsr, HEX);
  if (bfsr  & (1 << 3)) Serial.println(F("  -> IMPRECISE bus fault (BFAR may be stale/unreliable)"));
  if (bfsr  & (1 << 1)) Serial.println(F("  -> PRECISE bus fault (BFAR is the actual faulting access)"));
  if (bfsr  & (1 << 0)) Serial.println(F("  -> instruction bus fault (fetch)"));
  if (ufsr  & (1 << 0)) Serial.println(F("  -> UNDEFINSTR (illegal instruction)"));
  if (ufsr  & (1 << 8)) Serial.println(F("  -> UNALIGNED access"));
  if (mmfsr & (1 << 0)) Serial.println(F("  -> instruction MPU violation"));
  if (mmfsr & (1 << 1)) Serial.println(F("  -> data MPU violation"));
  Serial.println();
}

// ==================== Species labels ====================
// Order must match the training label encoding, which is NOT alphabetical.
// Taken from this exact run's classification_report_int8.txt (rows are
// emitted in label-index order) and cross-checked against
// src/MynaNet/MynaNet.ino -- both agree. Getting this wrong mislabels
// every prediction while still looking plausible, so don't "tidy" it into
// alphabetical order.
constexpr int NUM_CLASSES = 12;
const char *SPECIES_NAMES[NUM_CLASSES] = {
    "Coppersmith Barbet",
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

// ==================== Audio ====================
const int micPin            = A0;  // this bench's actual mic wiring
const int SAMPLE_RATE       = MYNA_SR;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // 96 000 B BSS

const int INPUT_SIZE = MYNA_N_MELS * MYNA_TIME_FRAMES;  // 64*300 = 19200

// Mel output as float before quantisation. 19200 floats = 75 KB BSS --
// needed because power_to_db(ref=np.max) requires the whole clip's peak
// before any value can be normalised, so a single streaming pass into the
// INT8 tensor isn't possible.
static float melFeatures[INPUT_SIZE];

// RMS gate, ADC units (int16 scale +-32767) -- skip inference on silent
// windows. Same starting point as the DrongoNet sketches; recalibrate for
// your mic/gain.
#define RMS_THRESHOLD 500.0f

// ==================== TensorFlow Lite ====================
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model            = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input  = nullptr;
TfLiteTensor *output = nullptr;

// 1.5 MB, matching src/MynaNet/MynaNet.ino's proven size for this model
// family. Actual usage is printed at boot so it can be right-sized later.
constexpr int kTensorArenaSize = 1500 * 1024;
uint8_t *tensor_arena = nullptr;  // from SDRAM.malloc() in setup()
}  // namespace

// ==================== CMSIS-DSP FFT + windows ====================
static arm_rfft_fast_instance_f32 fftInstance;

// Hann(400) zero-padded to 512 and centred, exactly as librosa's
// pad_center() does -- offset (512-400)/2 = 56. Periodic (fftbins=True)
// hann, matching librosa's get_window default.
__attribute__((section(".dtcmram")))
static float window512[MYNA_N_FFT];

__attribute__((section(".dtcmram")))
static float fftBuf[MYNA_N_FFT];

__attribute__((section(".dtcmram")))
static float powerSpectrum[MYNA_N_FFT_BINS];

// ===========================================================
void initMelFFT() {
  const int pad = (MYNA_N_FFT - MYNA_WIN_LENGTH) / 2;  // 56
  for (int i = 0; i < MYNA_N_FFT; i++) window512[i] = 0.0f;
  for (int n = 0; n < MYNA_WIN_LENGTH; n++) {
    window512[pad + n] =
        0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)MYNA_WIN_LENGTH);
  }
  arm_rfft_fast_init_f32(&fftInstance, MYNA_N_FFT);
}

// Reflect-pad index lookup, emulating librosa's center=True (which
// reflect-pads the signal by n_fft/2 before framing) without allocating a
// second padded copy of the 48 000-sample buffer.
static inline int reflectIndex(int idx, int n) {
  if (idx < 0) idx = -idx;
  if (idx >= n) idx = 2 * (n - 1) - idx;
  if (idx < 0) idx = 0;
  return idx;
}

// fastLog10f -- IEEE-754 exponent bit-trick approximation (~10 cycles vs
// ~150 for libm log10f(), error < 0.5 dB, negligible against MynaNet's
// own MYNA_DB_MIN/MAX clip range below). Same technique as
// ARGUS_Common.h's argus_fast_log10f() / the DrongoNet instrumented
// sketches' fastLog10f(), inlined here rather than #include-ing
// ARGUS_Common.h for the same include-conflict reason documented in
// h7_drongonet_m4_instrumented.ino's file header. Ported here per
// DRONGONET_SPARSE_MEL_BENCHMARK.md's finding (this sketch predates
// that finding and was still on plain log10f()).
static inline float fastLog10f(float x) {
  if (x <= 0.0f) return -100.0f;
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  return (exp + mant) * 0.30103f;
}

// ===========================================================
// Mel spectrogram -- mirrors 1o_fcn_epilogue.py compute_spec()
// ===========================================================
void computeMelSpectrogram(const int16_t *audio, int audioLength, float *melOut) {
  const float INV_32768 = 1.0f / 32768.0f;
  const int half = MYNA_N_FFT / 2;

  for (int t = 0; t < MYNA_TIME_FRAMES; t++) {
    const int start = t * MYNA_HOP_LENGTH - half;  // center=True

    for (int i = 0; i < MYNA_N_FFT; i++) {
      int idx = reflectIndex(start + i, audioLength);
      fftBuf[i] = (float)audio[idx] * INV_32768 * window512[i];
    }

    arm_rfft_fast_f32(&fftInstance, fftBuf, fftBuf, 0 /*forward*/);

    // power=2.0 -> |X[k]|^2, with DC and Nyquist packed in bins 0/1 by CMSIS
    powerSpectrum[0]    = fftBuf[0] * fftBuf[0];
    powerSpectrum[half] = fftBuf[1] * fftBuf[1];
    arm_cmplx_mag_squared_f32(&fftBuf[2], &powerSpectrum[1], half - 1);

    // Column-major in the model's (mel, time) layout: melOut[mel][t]
    for (int m = 0; m < MYNA_N_MELS; m++) {
      const float *P = &powerSpectrum[mynaMelFilters[m].startBin];
      const float *W = &mynaMelWeightFlat[mynaMelFilters[m].offset];
      const int width = mynaMelFilters[m].width;

      float acc = 0.0f;
      for (int k = 0; k < width; k++) acc += P[k] * W[k];

      melOut[m * MYNA_TIME_FRAMES + t] = acc;
    }
  }

  // power_to_db(ref=np.max): 10*log10(mel / max(mel)), floored the way
  // librosa does (top_db is not applied here -- the explicit clip below
  // to the training dB bounds subsumes it).
  float peak = 0.0f;
  for (int i = 0; i < INPUT_SIZE; i++) {
    if (melOut[i] > peak) peak = melOut[i];
  }
  if (peak < 1e-10f) peak = 1e-10f;
  const float invPeak = 1.0f / peak;

  const float range    = MYNA_DB_MAX - MYNA_DB_MIN;
  const float invRange = 1.0f / (range + 1e-8f);

  for (int i = 0; i < INPUT_SIZE; i++) {
    float v = melOut[i] * invPeak;
    if (v < 1e-10f) v = 1e-10f;
    float db = 10.0f * fastLog10f(v);
    if (db < MYNA_DB_MIN) db = MYNA_DB_MIN;
    if (db > MYNA_DB_MAX) db = MYNA_DB_MAX;
    melOut[i] = (db - MYNA_DB_MIN) * invRange;
  }
}

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

float   inputScale = 0.0f,  outputScale = 0.0f;
int32_t inputZp    = 0,     outputZp    = 0;

void topK(const int8_t *q, int n, int k, int *idx, float *prob,
          float scale, int32_t zp) {
  for (int i = 0; i < k; i++) { idx[i] = -1; prob[i] = -1.0f; }
  for (int i = 0; i < n; i++) {
    float p = (q[i] - zp) * scale;
    for (int s = 0; s < k; s++) {
      if (p > prob[s]) {
        for (int m = k - 1; m > s; m--) { prob[m] = prob[m-1]; idx[m] = idx[m-1]; }
        prob[s] = p; idx[s] = i;
        break;
      }
    }
  }
}

void setup() {
  // Enable the specific BusFault/UsageFault/MemManage handlers instead of
  // letting them all escalate to HardFault -- mbed_error_hook() (see
  // above) fires either way since mbed's own except.S routes all four
  // through it, but SCB->CFSR distinguishes the real cause regardless,
  // so this is cheap correctness, not load-bearing. Must happen before
  // anything that could fault.
  SCB->SHCSR |= SCB_SHCSR_MEMFAULTENA_Msk | SCB_SHCSR_BUSFAULTENA_Msk | SCB_SHCSR_USGFAULTENA_Msk;

  Serial.begin(115200);
  while (!Serial) { ; }
  delay(1000);

  pinMode(micPin, INPUT);
  analogReadResolution(12);  // board defaults to 10-bit; captureAudio()'s
                             // -2048/2048 scaling assumes true 12-bit
  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, HIGH);  // OFF (active-LOW)

  DWT_ENABLE();

  Serial.println(F("=== h7_mynanet_m7_instrumented: MynaNet 1o on M7 ==="));
  checkAndReportPreviousFault();

  Wire.begin();
  if (ina219.begin()) {
    ina219_ok = true;
    Serial.println(F("[OK]  INA219 found -- energy will be reported."));
  } else {
    Serial.println(F("[WARN] INA219 not found -- energy reporting skipped."));
  }

  SDRAM.begin();
  Serial.println(F("[OK]  SDRAM initialised."));

#if ENABLE_SDRAM_CACHE
  configureSdramCacheable();
  Serial.println(F("[OK]  SDRAM region marked cacheable (ENABLE_SDRAM_CACHE=1)."));
#endif

  uint8_t *raw = (uint8_t *)SDRAM.malloc(kTensorArenaSize + 16);
  if (!raw) {
    Serial.println(F("[ERR] SDRAM.malloc() failed for tensor arena."));
    while (1) { }
  }
  tensor_arena = (uint8_t *)(((uintptr_t)raw + 15u) & ~15u);
  Serial.print(F("[OK]  Arena "));
  Serial.print(kTensorArenaSize / 1024);
  Serial.print(F(" KB in SDRAM at 0x"));
  Serial.println((uint32_t)tensor_arena, HEX);

  model = tflite::GetModel(mynanet_1o);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Model load failed or schema mismatch."));
    while (1) { }
  }

  // Ops for 1o = 1j's MBV3 blocks + MatchboxNet FCN epilogue. Starts from
  // the 7 ops src/MynaNet/MynaNet.ino verified against 1j's flatbuffer,
  // plus ADD/PAD/RESHAPE/QUANTIZE/DEQUANTIZE headroom for the epilogue and
  // any explicit quantise boundary nodes. If AllocateTensors() reports a
  // missing op, add it here rather than switching to AllOpsResolver -- the
  // failure message names the opcode.
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddMean();
  resolver.AddPad();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("[ERR] AllocateTensors() failed -- missing op, or arena too small."));
    while (1) { }
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
  getTensorQuantParams(input,  &inputScale,  &inputZp);
  getTensorQuantParams(output, &outputScale, &outputZp);

  Serial.print(F("[OK]  arena used "));
  Serial.print((uint32_t)interpreter->arena_used_bytes() / 1024);
  Serial.print(F(" / "));
  Serial.print(kTensorArenaSize / 1024);
  Serial.println(F(" KB"));
  Serial.print(F("[OK]  input  type="));
  Serial.print((int)input->type);
  Serial.print(F(" bytes="));
  Serial.print(input->bytes);
  Serial.print(F(" scale="));
  Serial.print(inputScale, 8);
  Serial.print(F(" zp="));
  Serial.println(inputZp);
  Serial.print(F("[OK]  output type="));
  Serial.print((int)output->type);
  Serial.print(F(" bytes="));
  Serial.print(output->bytes);
  Serial.print(F(" scale="));
  Serial.print(outputScale, 8);
  Serial.print(F(" zp="));
  Serial.println(outputZp);

  if (input->bytes != (uint32_t)INPUT_SIZE) {
    Serial.print(F("[WARN] input->bytes != INPUT_SIZE ("));
    Serial.print(INPUT_SIZE);
    Serial.println(F(") -- mel geometry and model disagree."));
  }

  initMelFFT();
  Serial.println(F("[OK]  CMSIS-DSP RFFT + hann(400)@512 initialised."));
  Serial.println(F("[OK]  Ready.\n"));
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
    Serial.print(F("[GATE] Silent window -- RMS="));
    Serial.print(rms, 1);
    Serial.println(F(" -- inference skipped."));
    return;
  }

  float power_mW = ina219_ok ? ina219.getPower_mW() : 0.0f;

  // ── STAGE: mel spectrogram only ──────────────────────────────
  DWT_RESET();
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, melFeatures);
  uint32_t melCycles = DWT_CYCLES();
  float melUs = DWT_US(melCycles);

  // Quantise into the INT8 input tensor (not timed as either stage --
  // it's a fixed cost that belongs to neither mel nor inference).
  for (int i = 0; i < INPUT_SIZE; i++) {
    float scaled = melFeatures[i] / inputScale + inputZp;
    if (scaled >  127.0f) scaled =  127.0f;
    if (scaled < -128.0f) scaled = -128.0f;
    input->data.int8[i] = (int8_t)scaled;
  }

  // Arena is in SDRAM and the writes above went through the M7 D-cache;
  // flush before the kernels read it. Omitting this is a silent
  // wrong-data hazard, not a crash -- see src/MynaNet/MynaNet.ino.
  SCB_CleanInvalidateDCache();

  // ── STAGE: inference only ────────────────────────────────────
  DWT_RESET();
  TfLiteStatus status = interpreter->Invoke();
  uint32_t inferCycles = DWT_CYCLES();
  float inferUs = DWT_US(inferCycles);

  if (status != kTfLiteOk) {
    Serial.println(F("[ERR] Invoke() failed."));
    return;
  }

  int topIdx[3];
  float topProb[3];
  topK(output->data.int8, NUM_CLASSES, 3, topIdx, topProb, outputScale, outputZp);

  Serial.print(F("RMS: "));
  Serial.println(rms, 0);
  for (int i = 0; i < 3; i++) {
    Serial.print(F("  "));
    Serial.print(i + 1);
    Serial.print(F(". "));
    Serial.print((topIdx[i] >= 0 && topIdx[i] < NUM_CLASSES)
                     ? SPECIES_NAMES[topIdx[i]] : "?");
    Serial.print(F("  ->  "));
    Serial.print(topProb[i] * 100.0f, 2);
    Serial.println(F(" %"));
  }

  float totalUs = melUs + inferUs;
  Serial.print(F("  [latency] mel="));
  Serial.print(melUs, 1);
  Serial.print(F(" us  infer="));
  Serial.print(inferUs, 1);
  Serial.print(F(" us  total="));
  Serial.print(totalUs, 1);
  Serial.println(F(" us"));

  if (ina219_ok) {
    float eMel   = power_mW * (melUs   / 1000.0f);
    float eInfer = power_mW * (inferUs / 1000.0f);
    Serial.print(F("  [energy] power="));
    Serial.print(power_mW, 1);
    Serial.print(F(" mW  mel="));
    Serial.print(eMel, 1);
    Serial.print(F(" uJ  infer="));
    Serial.print(eInfer, 1);
    Serial.print(F(" uJ  total="));
    Serial.print(eMel + eInfer, 1);
    Serial.println(F(" uJ"));
  }

  Serial.println();
}
