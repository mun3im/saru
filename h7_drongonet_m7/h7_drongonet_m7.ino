// h7_drongonet_m7.ino — DrongoNet-nano, single-core M7, no RPC
// Model: DrongoNet-nano (INT8, 16x184 mel spectrogram -> softmax[no_bird,bird])
// Same DSP/inference pipeline as h7_drongonet/drongonet_m4 and
// src/DrongoNet/DrongoNet_Nano, but everything -- mic capture, mel,
// inference, and reporting -- runs synchronously on M7 with direct
// Serial access. No RPC, no paired M7 receiver sketch, no dual-core
// flash-split bookkeeping: just flash this one sketch and open the
// serial monitor. Meant as the easiest variant of this pipeline to
// bring up and debug when something in the mel/inference path itself
// is suspect, decoupled from any dual-core RPC questions.

#include <arm_math.h>  // CMSIS-DSP, bundled with the Portenta board package

#ifdef abs
#undef abs
#endif

#include "Chirale_TensorFlowLite.h"  // replaces the old Arduino_TensorFlowLite
                                     // (2.4.0-ALPHA, unmaintained since ~2021)
                                     // -- that library's micro_allocator never
                                     // populated tensor->quantization for this
                                     // model (confirmed: quantization.type=0 at
                                     // runtime for I/O tensors), which silently
                                     // broke inference (constant output
                                     // regardless of real input). Chirale is a
                                     // current TFLM snapshot, Portenta-supported.
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
const int micPin            = A0;  // matches this bench's actual mic wiring (h7_goertzel_rpc used A0 too)
const int SAMPLE_RATE       = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // 96 000 B BSS

// RMS energy gate, ADC units (int16 scale +-32767) -- skip inference on
// silent windows. Same starting point as DrongoNet_Nano; recalibrate for
// your mic/gain.
#define RMS_THRESHOLD 500.0f

// avgScore > this -> bird detected. Raised from the 0.6 default: measured
// aircon-hum baseline in this room reads P(bird) 0.62-0.76, overlapping the
// real bird-sound test's 0.68-0.77 -- 0.8 doesn't fully separate the two
// (nothing will, given that much overlap) but cuts obvious false positives.
// Nano has no published/locked threshold (see mun3im/drongonet's
// MODEL_REFERENCE.md); re-measure if the acoustic environment changes.
#define BIRD_SCORE_THRESHOLD 0.8f

// ==================== TensorFlow Lite ====================
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;

const tflite::Model *model            = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input  = nullptr;
TfLiteTensor *output = nullptr;

// DrongoNet-nano typically uses ~28-40 KB of arena (see
// src/DrongoNet/DrongoNet_Nano's profiling comments). M7 has no RPC/OpenAMP
// DTCM reservation to fight over (unlike M4 -- see drongonet_m4.ino), so
// there's no tight arena ceiling here; 64 KB gives comfortable headroom
// against M7's ~454 KB free RAM without needing to shave it close.
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ==================== CMSIS-DSP FFT instance + buffers ====================
static arm_rfft_fast_instance_f32 fftInstance;

// DTCM placement is safe here (M7-only, no RPC/OpenAMP conflict) -- see
// src/DrongoNet/DrongoNet_Nano for the same pattern.
__attribute__((section(".dtcmram")))
static float hannWindow[MEL_N_FFT];

__attribute__((section(".dtcmram")))
static float fftBuf[MEL_N_FFT];

__attribute__((section(".dtcmram")))
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

// No cross-window smoothing/state -- each 3s capture window is judged
// independently, so back-to-back calls in separate windows each register
// their own detection instead of needing the (now-removed) rolling average
// to dip back below threshold first.

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

      melRow[m] = 10.0f * log10f(acc + 1e-10f);
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
// Note on I/O dtype: this model's declared graph input/output tensors
// (`serving_default_input_1:0` and `StatefulPartitionedCall:0`, per a raw
// FlatBuffer schema dump) are FLOAT32, not INT8 -- confirmed by their
// TensorType()==0 (FLOAT32) with no quantization table attached. The
// INT8 tensors (scale=0.00392157/zp=-128 on the input side, matching
// mun3im/drongonet's MODEL_REFERENCE.md) sit *inside* the graph, between
// explicit QUANTIZE/DEQUANTIZE ops -- this is what a TF converter produces
// when full-integer quantization is used without also setting
// inference_input_type/inference_output_type to int8. TFLM's own
// QUANTIZE/DEQUANTIZE kernels handle that internal conversion using the
// model's own embedded per-op scale/zero_point; the sketch just needs to
// read/write plain floats at the graph boundary -- no manual
// quantize/dequantize math needed at all.
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

  Serial.println(F("=== h7_drongonet_m7: DrongoNet-nano, single-core M7 ==="));

  model = tflite::GetModel(drongonet_nano);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println(F("[ERR] Model load failed or schema mismatch."));
    while (1) { }
  }

  // Raw schema-level check -- bypasses the TFLM runtime allocator entirely,
  // reads straight from the embedded FlatBuffer to see whether quantization
  // metadata is actually present in drongonet-nano.h's byte array itself.
  {
    const auto *subgraph = model->subgraphs()->Get(0);
    const auto *tensors  = subgraph->tensors();
    Serial.print(F("[SCHEMA] tensor count: "));
    Serial.println(tensors->size());
    for (int i = 0; i < (int)tensors->size(); i++) {
      const auto *t = tensors->Get(i);
      const auto *q = t->quantization();
      Serial.print(F("  tensor["));
      Serial.print(i);
      Serial.print(F("] name="));
      Serial.print(t->name() ? t->name()->c_str() : "?");
      Serial.print(F(" type="));
      Serial.print((int)t->type());
      if (q && q->scale() && q->scale()->size() > 0) {
        Serial.print(F(" scale="));
        Serial.print(q->scale()->Get(0), 8);
        Serial.print(F(" zp="));
        Serial.print(q->zero_point() && q->zero_point()->size() > 0 ? q->zero_point()->Get(0) : 0);
      } else {
        Serial.print(F(" [NO QUANT IN SCHEMA]"));
      }
      Serial.println();
    }
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
    Serial.println(F("[ERR] AllocateTensors() failed -- bump kTensorArenaSize."));
    while (1) { }
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print(F("[OK]  input  type="));
  Serial.print((int)input->type);
  Serial.print(F(" bytes="));
  Serial.println(input->bytes);
  Serial.print(F("[OK]  output type="));
  Serial.print((int)output->type);
  Serial.print(F(" bytes="));
  Serial.println(output->bytes);

  Serial.print(F("[OK]  Arena used: "));
  Serial.print((uint32_t)interpreter->arena_used_bytes());
  Serial.print(F(" / "));
  Serial.print(kTensorArenaSize);
  Serial.println(F(" B"));

  initMelFFT();
  Serial.println(F("[OK]  CMSIS-DSP RFFT + Hann window initialised."));
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

  static float melFeatures[INPUT_SIZE];
  computeMelSpectrogram(audioBuffer, BUFFER_SIZE, melFeatures);

  float melMin = melFeatures[0], melMax = melFeatures[0];
  for (int i = 0; i < INPUT_SIZE; i++) {
    if (melFeatures[i] < melMin) melMin = melFeatures[i];
    if (melFeatures[i] > melMax) melMax = melFeatures[i];
  }

  // Model's declared graph input is FLOAT32 (see the I/O dtype note above)
  // -- write mel features straight in, no manual quantization.
  memcpy(input->data.f, melFeatures, INPUT_SIZE * sizeof(float));

  Serial.print(F("  [diag] melFeatures min/max: "));
  Serial.print(melMin, 6);
  Serial.print(F(" / "));
  Serial.println(melMax, 6);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("[ERR] Invoke() failed."));
    return;
  }

  // Model's declared graph output is FLOAT32 too -- softmax[no_bird, bird],
  // already dequantized internally by the model's own DEQUANTIZE op.
  float prob_no_bird = output->data.f[0];
  float prob_bird    = output->data.f[1];

  Serial.print(F("  [raw] P(no_bird)="));
  Serial.print(prob_no_bird, 6);
  Serial.print(F(" P(bird)="));
  Serial.println(prob_bird, 6);

  Serial.print(F("RMS: "));
  Serial.print(rms, 0);
  Serial.print(F(" | P(bird): "));
  Serial.println(prob_bird, 4);

  // Each window judged on its own -- no cross-window state, so back-to-back
  // calls in separate 3s windows each print their own BIRD.
  if (prob_bird > BIRD_SCORE_THRESHOLD) {
    Serial.println(F("BIRD"));
    digitalWrite(LEDR, LOW);   // ON
    delay(500);
    digitalWrite(LEDR, HIGH);  // OFF
  }
}
