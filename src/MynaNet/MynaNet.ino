#include <Arduino.h>

#ifdef abs
#undef abs
#endif
#include <cmath>

// ── Memory Management ─────────────────────────────────────────────────
#include "SDRAM.h"
#include <mbed.h>
#include <mbed_stats.h>

// ── TensorFlow Lite Micro ─────────────────────────────────────────────
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ── Shared ARGUS helpers ────────────────────────────────────────────────
// DWT timing macros and the system storage breakdown -- see
// libraries/ARGUS_Common/src/ARGUS_Common.h. Install this library (copy
// libraries/ARGUS_Common into your Arduino libraries folder, or open the
// .ino via the repo checkout) before compiling this sketch.
#include <ARGUS_Common.h>

// ── INA219 Power Monitor ──────────────────────────────────────────────
// Hardware: INA219 I2C breakout wired to Portenta SDA/SCL.
// Tracks bus voltage, current, power per inference cycle, and
// accumulates total energy (mWh) across the session.
// If INA219 is absent the sketch continues — power lines print 0.
#include <Adafruit_INA219.h>
#include <Wire.h>

Adafruit_INA219 ina219;
bool ina219_ok = false;
float total_energy_mWh = 0.0f;
unsigned long lastPowerMs = 0;

// ── Model + Sample ────────────────────────────────────────────────────
#include "MynaNet_1j.h"
#include "Spotted_Dove_mel.h" // declares: const int8_t Spotted_Dove_mel[19200]

// ── Compile-time sanity check ─────────────────────────────────────────
// 64 mel bins × 300 time frames × 1 channel = 19 200 elements
static_assert(sizeof(Spotted_Dove_mel) == 64 * 300 * 1,
              "Spotted_Dove_mel size does not match model input [1,64,300,1]");

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

// ── TFLite globals ────────────────────────────────────────────────────
const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// ── Arena in SDRAM (1.5 MB)  ────────────────────────────────────────────
constexpr int kTensorArenaSize = 1500 * 1024;
uint8_t *tensor_arena = nullptr;

// ── Op Resolver  ──────────────────────────────────────────────────────
// Template argument = exact number of Add() calls below (must equal 7).
// Opcodes confirmed by flatbuffer parse + cross-checked against the
// installed library's schema_generated.h enum (older numbering).
static tflite::MicroMutableOpResolver<7> resolver;

// DWT_ENABLE()/DWT_RESET()/DWT_CYCLES()/DWT_US() now live in
// ARGUS_Common.h -- see the #include near the top of this file.

// ── Power helper ──────────────────────────────────────────────────────
struct PowerSample {
  float voltage_V;
  float current_mA;
  float power_mW;
};

PowerSample readPower() {
  PowerSample ps = {0.0f, 0.0f, 0.0f};
  if (!ina219_ok)
    return ps;
  ps.voltage_V = ina219.getBusVoltage_V();
  ps.current_mA = ina219.getCurrent_mA();
  ps.power_mW = ina219.getPower_mW();
  return ps;
}

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

// ── Memory Profiling Helpers ───────────────────────────────────────────
// printSystemStorage() now lives in ARGUS_Common.h as
// argus_print_system_storage() -- see the #include above.

void printFirmwareRAM() {
  Serial.println(F("\n╔══════════════════════════════════════════╗"));
  Serial.println(F("║      FIRMWARE RAM BREAKDOWN              ║"));
  Serial.println(F("╚══════════════════════════════════════════╝"));

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

// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000)
    ;
  Serial.println(F("\n=== Portenta H7  MynaNet  DS-CNN  v5 ==="));
  Serial.println(F("    MicroMutableOpResolver (7 ops, schema-verified)"));
  Serial.println(F("    INA219 power / energy monitoring enabled"));
  Serial.println(F("========================================\n"));

  // ── Enable DWT cycle counter ────────────────────────────────────────
  DWT_ENABLE();

  // ── INA219 initialisation ────────────────────────────────────────────
  Wire.begin();
  if (ina219.begin()) {
    ina219_ok = true;
    lastPowerMs = millis();
    Serial.println(F("[OK]  INA219 power monitor initialised."));
  } else {
    Serial.println(F("[WARN] INA219 not found — power data unavailable."));
    Serial.println(F("       Continuing without power metrics."));
  }

  // ── 1. Allocate tensor arena from SDRAM ────────────────────────────
  SDRAM.begin();
  void *raw = SDRAM.malloc(kTensorArenaSize + 16);
  if (!raw) {
    Serial.println(F("FATAL: SDRAM.malloc failed — check SDRAM.begin()"));
    while (1)
      ;
  }
  // Align to 16 bytes (CMSIS-NN SIMD requirement)
  tensor_arena = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(raw) + 15) & ~15UL);
  Serial.print(F("Arena: "));
  Serial.print(kTensorArenaSize / 1024);
  Serial.println(F(" KB allocated in SDRAM"));

  // ── 2. Register the 7 ops — matched against THIS library's enum ──────
  // Flatbuffer numeric opcodes are identical across schema versions;
  // only the C++ enum NAMES shifted between old and new TFLite schemas.
  resolver.AddConv2D();          // flatbuf opcode  3 = CONV_2D
  resolver.AddDepthwiseConv2D(); // flatbuf opcode  4 = DEPTHWISE_CONV_2D
  resolver.AddMaxPool2D(); // flatbuf opcode 17 = MAX_POOL_2D  (older enum)
  resolver.AddMul();       // flatbuf opcode 18 = MUL (v2 — kernel handles both)
  resolver.AddFullyConnected(); // flatbuf opcode  9 = FULLY_CONNECTED
  resolver.AddSoftmax(); // flatbuf opcode 25 = SOFTMAX       (older enum)
  resolver.AddMean();    // flatbuf opcode 40 = MEAN           (older enum)

  // ── 3. Load model ──────────────────────────────────────────────────
  g_model = tflite::GetModel(MynaNet_1j);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("FATAL: schema mismatch — model="));
    Serial.print(g_model->version());
    Serial.print(F("  runtime="));
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1)
      ;
  }
  Serial.println(F("Model loaded OK"));

  // ── 4. Build interpreter ───────────────────────────────────────────
  static tflite::MicroInterpreter static_interp(g_model, resolver, tensor_arena,
                                                kTensorArenaSize);
  interpreter = &static_interp;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println(F("FATAL: AllocateTensors() failed."));
    Serial.println(F("  Possible causes:"));
    Serial.println(
        F("  a) Op missing from resolver — add the missing AddXxx()"));
    Serial.println(F("  b) Arena too small — increase kTensorArenaSize"));
    while (1)
      ;
  }

  // Print arena usage (useful for right-sizing kTensorArenaSize later)
  Serial.print(F("Arena used: "));
  Serial.print(interpreter->arena_used_bytes() / 1024);
  Serial.println(F(" KB"));

  // ── 5. Cache tensor pointers ───────────────────────────────────────
  input = interpreter->input(0);
  output = interpreter->output(0);

  // ── 6. Verify tensor shapes and quantisation params ────────────────
  Serial.print(F("Input  shape: ["));
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1)
      Serial.print(',');
  }
  Serial.println(F("]  (expected [1,64,300,1])"));
  Serial.print(F("Input  type : "));
  Serial.println(input->type); // 9 = kTfLiteInt8
  Serial.print(F("Input  scale= "));
  Serial.print(input->params.scale, 6);
  Serial.print(F("  zp= "));
  Serial.println(input->params.zero_point);

  Serial.print(F("Output shape: ["));
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1)
      Serial.print(',');
  }
  Serial.println(F("]  (expected [1,12])"));
  Serial.print(F("Output scale= "));
  Serial.print(output->params.scale, 6);
  Serial.print(F("  zp= "));
  Serial.println(output->params.zero_point);

  // ── 7. Print memory breakdown ──────────────────────────────────────
  argus_print_system_storage();
  printFirmwareRAM();

  Serial.println(F("\nInit complete — starting inference loop"));
}

// ─────────────────────────────────────────────────────────────────────
void loop() {
  // ── A. Power snapshot BEFORE inference ─────────────────────────────
  PowerSample ps_before = readPower();

  unsigned long t0 = micros();

  // ── B. Load mel spectrogram into input tensor ──────────────────────
  memcpy(input->data.int8, Spotted_Dove_mel, input->bytes);

  // Flush D-cache: Cortex-M7 has a Harvard cache; tensor_arena lives in
  // SDRAM.  If the M7 wrote the memcpy through its D-cache but CMSIS-NN
  // reads SDRAM directly via DMA/AHB, the cache must be cleaned first.
  SCB_CleanInvalidateDCache();
  unsigned long t1 = micros();

  // ── C. Run inference (DWT-timed) ───────────────────────────────────
  DWT_RESET();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("ERROR: Invoke() failed — check resolver ops"));
    return;
  }
  uint32_t infer_cycles = DWT_CYCLES();
  unsigned long t2 = micros();

  // ── D. Top-3 post-processing ───────────────────────────────────────
  int top_idx[3];
  float top_prob[3];
  topK(output, NUM_CLASSES, 3, top_idx, top_prob);
  unsigned long t3 = micros();

  // ── E. Power snapshot AFTER inference ──────────────────────────────
  PowerSample ps_after = readPower();

  // Accumulate energy (trapezoidal average of before/after power).
  if (ina219_ok) {
    unsigned long now = millis();
    float dt_hours = (now - lastPowerMs) / 3600000.0f;
    lastPowerMs = now;
    float avg_power_mW = (ps_before.power_mW + ps_after.power_mW) * 0.5f;
    if (avg_power_mW > 0.0f)
      total_energy_mWh += avg_power_mW * dt_hours;
  }

  // ── F. Print results ───────────────────────────────────────────────
  Serial.println(F("\n╔══════════════════════════════╗"));
  Serial.println(F("║   MynaNet — Top 3 Prediction ║"));
  Serial.println(F("╚══════════════════════════════╝"));
  for (int i = 0; i < 3; i++) {
    Serial.print(i + 1);
    Serial.print(F(". "));
    Serial.print((top_idx[i] >= 0 && top_idx[i] < NUM_CLASSES)
                     ? SPECIES_NAMES[top_idx[i]]
                     : "Unknown");
    Serial.print(F("  →  "));
    Serial.print(top_prob[i] * 100.0f, 2);
    Serial.println(F(" %"));
  }

  // ── G. Latency breakdown ───────────────────────────────────────────
  Serial.println(F("─────────── Latency ────────────"));
  Serial.print(F("  Load  : "));
  Serial.print((t1 - t0) / 1000.0f, 2);
  Serial.println(F(" ms"));
  Serial.print(F("  Infer : "));
  Serial.print((t2 - t1) / 1000.0f, 2);
  Serial.println(F(" ms"));
  Serial.print(F("  Post  : "));
  Serial.print((t3 - t2) / 1000.0f, 2);
  Serial.println(F(" ms"));
  Serial.print(F("  Total : "));
  Serial.print((t3 - t0) / 1000.0f, 2);
  Serial.println(F(" ms"));
  Serial.print(F("  DWT   : "));
  Serial.print(DWT_US(infer_cycles), 1);
  Serial.println(F(" µs (inference only)"));

  // ── H. Power & Energy Breakdown ────────────────────────────────────
  // Single post-inference INA219 snapshot used as the active-load proxy.
  // Cortex-M7 @ 480 MHz busy-waits at 100% duty cycle through all stages,
  // so instantaneous power is representative of the compute load.
  // E(µJ) = P(mW) × t(ms)   [mW × ms = µW·s = µJ]
  //
  // Stage definitions (from µs timestamps already captured above):
  //   Mel load  : t0 → t1  (memcpy + SCB_CleanInvalidateDCache)
  //   Inference : t1 → t2  (Invoke)
  //   Post-proc : t2 → t3  (topK decode — tiny, included in total)
  //   Total     : t0 → t3  (full active cycle)
  Serial.println(F("\n─────────── Power & Energy ──────────────"));
  if (ina219_ok) {
    // Convert µs durations → ms for E = P × t
    float mel_load_ms = (float)(t1 - t0) * 1e-3f;   // memcpy + cache flush
    float infer_ms    = (float)(t2 - t1) * 1e-3f;   // inference
    float total_ms    = (float)(t3 - t0) * 1e-3f;   // full active cycle

    float pwr_mW = ps_after.power_mW;

    float energy_mel_uJ   = pwr_mW * mel_load_ms;   // Mel load
    float energy_infer_uJ = pwr_mW * infer_ms;      // Inference
    float energy_total_uJ = pwr_mW * total_ms;      // Total active cycle

    Serial.print(F("  Instantaneous  : "));
    Serial.print(pwr_mW, 2);
    Serial.print(F(" mW  ("));
    Serial.print(ps_after.voltage_V, 2);
    Serial.print(F(" V, "));
    Serial.print(ps_after.current_mA, 2);
    Serial.println(F(" mA)"));

    Serial.println(F("  [Energy — current cycle]"));
    Serial.print(F("    Mel load (memcpy+cache): "));
    Serial.print(energy_mel_uJ, 2);
    Serial.println(F(" µJ"));
    Serial.print(F("    Inference              : "));
    Serial.print(energy_infer_uJ, 2);
    Serial.println(F(" µJ"));
    Serial.print(F("    Total (active cycle)   : "));
    Serial.print(energy_total_uJ, 2);
    Serial.println(F(" µJ"));
    Serial.print(F("  Session cumulative       : "));
    Serial.print(total_energy_mWh, 6);
    Serial.println(F(" mWh"));
    Serial.println(F("─────────────────────────────────────────"));
  } else {
    Serial.println(F("  [no INA219 — power data unavailable]"));
    Serial.println(F("─────────────────────────────────────────"));
  }
  Serial.println(F("────────────────────────────────\n"));

  delay(200);
}
