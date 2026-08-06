// goertzel_m4_instrumented.ino — Goertzel-filter latency bench on M4, RPC
// to M7. Same DWT-cycle-accurate methodology as
// h7_drongonet_m4_instrumented, for direct latency comparison against
// DrongoNet's mel+inference figures on the same core.
//
// Algorithm/coefficients are the calibrated ones from
// h7_goertzel_rpc/goertzel_m4/goertzel_m4.ino (Fs=16kHz, 16-sample block,
// 3kHz centre -- see argus_h7_goertzel_calibration memory for the tuning
// history). Two differences from that live-detector sketch, both purely
// about making this a fair timing bench rather than a live field
// detector:
//   1. Compute is timed against a pre-captured 3s/48000-sample buffer,
//      split into 3000 consecutive non-overlapping 16-sample blocks, NOT
//      the live analogRead()-paced loop the original uses -- the original
//      sketch's ~1ms-per-block wall time is dominated by busy-waiting for
//      the next real-time sample tick, not compute, so timing it directly
//      would measure the sample rate, not the filter. This mirrors how
//      the DrongoNet instrumented sketches separate captureAudio() (timed
//      only for its own sake) from the DWT-timed compute region.
//   2. 12-bit ADC (analogReadResolution(12), midpoint 2048) instead of
//      the original's implicit 10-bit assumption (midpoint 512) -- see
//      h7_drongonet_m4_instrumented's own comment for why 10-bit was the
//      wrong assumption on this board. This shifts reported peak-
//      magnitude values by roughly 4x vs. the calibrated MAG_THRESHOLD=150
//      (tuned at 10-bit) -- NOT re-calibrated here, since this sketch
//      measures latency only; peak magnitude / above-threshold count are
//      reported for sanity/interest, not as a validated detection
//      decision.
//
// M4 has no direct USB Serial -- relays results to M7 via RPC, same
// pattern as h7_drongonet_m4_instrumented.

#include "RPC.h"

// DWT cycle-counter macros -- see h7_drongonet_m4_instrumented for why
// these are inlined rather than pulled from a shared header.
#define DWT_ENABLE()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; \
                           DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while(0)
#define DWT_RESET()   do { DWT->CYCCNT = 0; } while(0)
#define DWT_CYCLES()  (DWT->CYCCNT)
#define DWT_US(cyc)   ((float)(cyc) / 480.0f)   // us at 480 MHz -- both cores run at the same clock here

// ==================== Audio settings (matches DrongoNet instrumented sketches) ====================
const int micPin            = A0;
const int SAMPLE_RATE       = 16000;
const int AUDIO_DURATION_MS = 3000;
const int BUFFER_SIZE       = (SAMPLE_RATE * AUDIO_DURATION_MS) / 1000;  // 48000

int16_t audioBuffer[BUFFER_SIZE];  // centered raw ADC counts, NOT rescaled to int16 full range
                                    // (see file header note 2 -- keeps magnitude on the
                                    // calibration's original ADC-count scale, mod the 10-vs-12-bit shift)

// ==================== Goertzel filter (calibrated params, see file header) ====================
const float SAMPLE_RATE_HZ = 16000.0f;
const int   BLOCK_SIZE     = 16;
const float TARGET_FREQ_HZ = 3000.0f;
const float MAG_THRESHOLD  = 150.0f;  // NOT re-calibrated for 12-bit -- see file header note 2

const float OMEGA = 2.0f * PI * TARGET_FREQ_HZ / SAMPLE_RATE_HZ;
const float COEFF = 2.0f * cosf(OMEGA);

const int NUM_BLOCKS = BUFFER_SIZE / BLOCK_SIZE;  // 3000

// Characterization run: average window-compute latency over a fixed
// number of full 3s-buffer scans, then report the average once and stop
// -- same convention as h7_drongonet_m4_instrumented's CHARACTERIZE_COUNT.
#define CHARACTERIZE_COUNT 100

static int      characterizeCount   = 0;
static uint32_t characterizeSumUs   = 0;
static bool     characterizeDone    = false;

// ===========================================================
// Audio capture (busy-wait, precise 16 kHz) -- same pattern as the
// DrongoNet instrumented sketches, but stores centered raw ADC counts
// (not rescaled to int16 full range) to match this filter's calibration.
// ===========================================================
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

// ===========================================================
// Goertzel compute over one full captured window -- 3000 consecutive
// non-overlapping 16-sample blocks, identical per-block math to
// goertzel_m4.ino's runGoertzelBlock(), just fed from a buffer instead of
// live-paced analogRead() calls (see file header note 1).
// ===========================================================
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
  pinMode(micPin, INPUT);
  analogReadResolution(12);  // this board defaults to 10-bit -- see file header note 2
  RPC.begin();

  DWT_ENABLE();

  // M7 boots M4 from inside its own RPC.begin(), then binds its handlers
  // -- same handshake-race concern as h7_drongonet_m4_instrumented.
  delay(2000);
  RPC.call("reportReady", (int)NUM_BLOCKS, (int)BLOCK_SIZE);
}

void loop() {
  if (characterizeDone) {
    delay(5000);  // characterization complete -- idle
    return;
  }

  captureAudio();

  int above = 0;
  DWT_RESET();
  float peakMag = runGoertzelWindow(audioBuffer, NUM_BLOCKS, &above);
  uint32_t windowCycles = DWT_CYCLES();
  float windowUs = DWT_US(windowCycles);
  int windowUsInt = (int)(windowUs + 0.5f);

  RPC.call("reportGoertzel", (int)peakMag, above, windowUsInt);

  characterizeSumUs += (uint32_t)windowUsInt;
  characterizeCount++;
  if (characterizeCount >= CHARACTERIZE_COUNT && !characterizeDone) {
    characterizeDone = true;
    float avgUs = (float)characterizeSumUs / characterizeCount;
    RPC.call("reportGoertzelAverage", characterizeCount, (int)(avgUs + 0.5f));
  }
}
