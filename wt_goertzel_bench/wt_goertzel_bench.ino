// wt_goertzel_bench.ino -- Goertzel filter latency on Wio Terminal
// (Microchip ATSAMD51P19A, Cortex-M4F @ 120MHz), for cross-platform
// comparison against Portenta H7 M4/M7
// (h7_goertzel_rpc/goertzel_m4_instrumented, h7_goertzel_m7_instrumented)
// -- see GOERTZEL_VS_DRONGONET_LATENCY.md.
//
// Same algorithm/coefficients as the Portenta benches: single
// fixed-frequency Goertzel filter (Fs=16kHz, 16-sample block, 3kHz
// centre -- the calibrated h7_goertzel_rpc params, see
// argus_h7_goertzel_calibration memory), scanned as 3000 consecutive
// non-overlapping 16-sample blocks over one 3s/48000-sample window.
//
// Synthetic input, same reasoning as wt_drongonet_micro_bench.ino and
// every other cross-chip latency sketch in this repo: Goertzel's
// per-block work (fixed MACs + one magnitude calc) is entirely
// content-independent for a fixed block/window size, so a deterministic
// synthetic tone gives the same latency real captured audio would. No
// live-mic capture sketch exists for this board (same scope note as
// WIO_TERMINAL_DRONGONET_LATENCY.md -- "out of scope for this pass").
//
// Timing via micros(), same convention as wt_drongonet_micro_bench.ino
// (not the DWT cycle-counter macros the Portenta benches use -- no
// established DWT convention on this board yet, and micros()'s 1us
// resolution is more than enough at this filter's few-ms-per-window
// scale).
//
// No CMSIS-DSP / TFLite needed at all -- Goertzel is plain scalar
// arithmetic (1 multiply-add recurrence per sample, one sqrt per block),
// portable as-is, unlike DrongoNet's FFT+CNN pipeline.

#include <TFT_eSPI.h>  // bundled with Seeeduino:samd, see wt_drongonet_micro_bench.ino's header

// ==================== Goertzel filter (calibrated params, matches Portenta benches) ====================
constexpr float SAMPLE_RATE_HZ = 16000.0f;
constexpr int   BLOCK_SIZE     = 16;
constexpr float TARGET_FREQ_HZ = 3000.0f;
constexpr float MAG_THRESHOLD  = 150.0f;  // NOT re-calibrated for this board -- timing bench only,
                                           // see the Portenta instrumented sketches' file headers

const float OMEGA = 2.0f * PI * TARGET_FREQ_HZ / SAMPLE_RATE_HZ;
const float COEFF = 2.0f * cosf(OMEGA);

constexpr int WINDOW_SAMPLES = 48000;  // 3s @ 16kHz, matches the Portenta benches' capture window
constexpr int NUM_BLOCKS     = WINDOW_SAMPLES / BLOCK_SIZE;  // 3000
constexpr int NUM_RUNS       = 100;  // matches the Portenta M4/M7 Goertzel benches' 100-window characterization

static int16_t audioBuffer[WINDOW_SAMPLES];

void fillSyntheticAudio(int16_t *buf, int n) {
  for (int i = 0; i < n; i++) {
    buf[i] = (int16_t)(8000.0f * sinf((float)i * 0.017f));  // same synthetic tone as wt_drongonet_micro_bench.ino
  }
}

// Same per-block math as goertzel_m4_instrumented.ino's runGoertzelWindow().
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

// ==================== LCD status (see wt_drongonet_micro_bench.ino) ====================
TFT_eSPI tft = TFT_eSPI();

void showStatus(const String &line1, const String &line2 = "") {
  tft.fillScreen(TFT_BLACK);
  tft.setCursor(10, 10, 4);
  tft.setTextColor(TFT_CYAN, TFT_BLACK);
  tft.println("Goertzel bench");
  tft.setCursor(10, 60, 4);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.println(line1);
  if (line2.length()) {
    tft.setCursor(10, 100, 2);
    tft.setTextColor(TFT_YELLOW, TFT_BLACK);
    tft.println(line2);
  }
}

void setup() {
  Serial.begin(115200);
  uint32_t waitStart = millis();
  while (!Serial && millis() - waitStart < 5000) { }
  delay(500);

  Serial.println(F("=== wt_goertzel_bench: Goertzel filter on Wio Terminal (SAMD51) ==="));

  tft.init();
  tft.setRotation(3);
  showStatus("Booting...", "Init synthetic audio");

  fillSyntheticAudio(audioBuffer, WINDOW_SAMPLES);
  Serial.print(F("[OK]  blocks/window: "));
  Serial.print(NUM_BLOCKS);
  Serial.print(F("  block size: "));
  Serial.println(BLOCK_SIZE);
  Serial.println(F("[OK]  Running Goertzel benchmark now.\n"));

  static uint32_t windowTimes[NUM_RUNS];

  for (int i = 0; i < NUM_RUNS; i++) {
    // LCD update -- outside the timed region, so it doesn't pollute the
    // measured latency (same convention as wt_drongonet_micro_bench.ino).
    if (i % 10 == 0) {
      showStatus("Running benchmark", "run " + String(i + 1) + " / " + String(NUM_RUNS));
    }

    int above = 0;
    uint32_t t0 = micros();
    float peakMag = runGoertzelWindow(audioBuffer, NUM_BLOCKS, &above);
    uint32_t t1 = micros();
    windowTimes[i] = t1 - t0;

    Serial.print(F("  run "));
    Serial.print(i);
    Serial.print(F(": peakMag="));
    Serial.print((int)peakMag);
    Serial.print(F("  blocksAboveThreshold="));
    Serial.print(above);
    Serial.print(F("  window="));
    Serial.print(windowTimes[i]);
    Serial.println(F("us"));
  }

  uint32_t sum = 0, mn = UINT32_MAX, mx = 0;
  for (int i = 0; i < NUM_RUNS; i++) {
    sum += windowTimes[i];
    if (windowTimes[i] < mn) mn = windowTimes[i];
    if (windowTimes[i] > mx) mx = windowTimes[i];
  }
  float avg = (float)sum / (float)NUM_RUNS;

  Serial.println();
  Serial.print(F("[GOERTZEL] runs=")); Serial.print(NUM_RUNS);
  Serial.print(F("  avg=")); Serial.print(avg, 1);
  Serial.print(F(" us  min=")); Serial.print(mn);
  Serial.print(F(" us  max=")); Serial.print(mx);
  Serial.println(F(" us"));

  Serial.println(F("\n=== Goertzel Wio Terminal characterization complete ==="));
  showStatus("DONE (" + String(NUM_RUNS) + " runs)", "avg=" + String((uint32_t)avg) + "us");
}

void loop() { delay(1000); }
