// h7_goertzel_m7_instrumented.ino — Goertzel-filter latency bench,
// single-core M7, no RPC, direct USB Serial. M7-side counterpart of
// h7_goertzel_rpc/goertzel_m4_instrumented -- same DWT-cycle-accurate
// methodology, same algorithm/coefficients, same 3s/48000-sample window
// split into 3000 consecutive 16-sample blocks. See that sketch's file
// header for the full rationale (buffer-replay vs. live-paced timing,
// 12-bit ADC vs. the original calibration's 10-bit assumption).
//
// Pattern matches h7_drongonet_m7 (standalone single-core, no RPC) rather
// than h7_drongonet_m7_instrumented's multi-model sweep -- there's only
// one filter here, so no per-loop model switching needed.

// DWT cycle-counter macros -- see h7_drongonet_m7_instrumented for why
// these are inlined rather than pulled from a shared header.
#define DWT_ENABLE()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; \
                           DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while(0)
#define DWT_RESET()   do { DWT->CYCCNT = 0; } while(0)
#define DWT_CYCLES()  (DWT->CYCCNT)
#define DWT_US(cyc)   ((float)(cyc) / 480.0f)   // us at 480 MHz

// ==================== Audio settings ====================
const int micPin            = A0;
const int SAMPLE_RATE       = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // centered raw ADC counts, see goertzel_m4_instrumented's header

// ==================== Goertzel filter (calibrated params) ====================
const float SAMPLE_RATE_HZ = 16000.0f;
const int   BLOCK_SIZE     = 16;
const float TARGET_FREQ_HZ = 3000.0f;
const float MAG_THRESHOLD  = 150.0f;  // NOT re-calibrated for 12-bit -- see goertzel_m4_instrumented's header

const float OMEGA = 2.0f * PI * TARGET_FREQ_HZ / SAMPLE_RATE_HZ;
const float COEFF = 2.0f * cosf(OMEGA);

const int NUM_BLOCKS = BUFFER_SIZE / BLOCK_SIZE;  // 3000

#define CHARACTERIZE_COUNT 100

static int      characterizeCount = 0;
static uint32_t characterizeSumUs = 0;
static bool     characterizeDone  = false;

void captureAudio() {
  unsigned long sampleInterval = 1000000UL / SAMPLE_RATE;
  unsigned long lastSample = micros();
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (micros() - lastSample < sampleInterval) {
    }
    lastSample = micros();
    audioBuffer[i] = (int16_t)(analogRead(micPin) - 2048);
  }
}

float runGoertzelWindow(const int16_t *audio, int numBlocks, int *blocksAboveThreshold) {
  float peakMag = 0.0f;
  int   above   = 0;

  for (int b = 0; b < numBlocks; b++) {
    const int16_t *block = &audio[b * BLOCK_SIZE];
    float s1 = 0.0f, s2 = 0.0f;
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float x  = (float)block[n];
      float s0 = x + COEFF * s1 - s2;
      s2 = s1;
      s1 = s0;
    }
    float real = s1 - s2 * cosf(OMEGA);
    float imag = s2 * sinf(OMEGA);
    float mag  = sqrtf(real * real + imag * imag);

    if (mag > peakMag) peakMag = mag;
    if (mag > MAG_THRESHOLD) above++;
  }

  *blocksAboveThreshold = above;
  return peakMag;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  delay(1000);

  pinMode(micPin, INPUT);
  analogReadResolution(12);  // this board defaults to 10-bit

  DWT_ENABLE();

  Serial.println(F("=== h7_goertzel_m7_instrumented: Goertzel filter on M7 ==="));
  Serial.print(F("[OK]  blocks/window: "));
  Serial.print(NUM_BLOCKS);
  Serial.print(F("  block size: "));
  Serial.println(BLOCK_SIZE);
  Serial.println(F("[OK]  Ready.\n"));
}

void loop() {
  if (characterizeDone) {
    delay(5000);
    return;
  }

  captureAudio();

  int above = 0;
  DWT_RESET();
  float peakMag = runGoertzelWindow(audioBuffer, NUM_BLOCKS, &above);
  uint32_t windowCycles = DWT_CYCLES();
  float windowUs = DWT_US(windowCycles);
  int windowUsInt = (int)(windowUs + 0.5f);

  Serial.print(F("[GOERTZEL] peakMag="));
  Serial.print((int)peakMag);
  Serial.print(F("  blocksAboveThreshold="));
  Serial.print(above);
  Serial.print(F("  window="));
  Serial.print(windowUsInt);
  Serial.println(F(" us"));

  characterizeSumUs += (uint32_t)windowUsInt;
  characterizeCount++;
  if (characterizeCount >= CHARACTERIZE_COUNT && !characterizeDone) {
    characterizeDone = true;
    float avgUs = (float)characterizeSumUs / characterizeCount;
    Serial.println();
    Serial.println(F("=== Goertzel M7 characterization complete ==="));
    Serial.print(F("[AVG over ")); Serial.print(characterizeCount); Serial.println(F(" windows]"));
    Serial.print(F("  avg window: ")); Serial.print((int)(avgUs + 0.5f)); Serial.println(F(" us"));
  }
}
