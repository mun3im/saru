#include "RPC.h"

// Single Goertzel filter, widened for broader species coverage. The
// 64-sample/Fs-4-centred version (matching AudioMoth's default slider
// position) measurably detected a real ebird.com playback (idle ~65-99,
// signal up to 6063 against a 5000 threshold), but its ~250 Hz-wide
// passband at 4 kHz barely responded to content around 2 kHz (peak
// magnitude ~200, well under threshold). Dropping to a 16-sample window
// (AudioMoth's narrowest/widest-passband option) and re-centring lower
// trades selectivity for coverage: Fs/N = 16000/16 = 1000 Hz per bin
// (Q = centre/bandwidth = 3), which should respond well anywhere roughly
// 2-4 kHz instead of only right at 4 kHz.
const int micPin = A0;

const float SAMPLE_RATE_HZ = 16000.0f;  // requested Fs
const int   BLOCK_SIZE     = 16;        // AudioMoth's narrowest window option -> widest passband
const float TARGET_FREQ_HZ = 3000.0f;   // re-centred lower to cover the ~2 kHz content that was being missed
const float MAG_THRESHOLD  = 400.0f;    // measured: idle ~42-96, signal ~249-488

const int ADC_MIDPOINT = 512;  // remove DC bias (10-bit ADC)

const float OMEGA = 2.0f * PI * TARGET_FREQ_HZ / SAMPLE_RATE_HZ;
const float COEFF = 2.0f * cosf(OMEGA);

const unsigned long REPORT_INTERVAL_MS = 1000;  // how often to report the peak magnitude, for MAG_THRESHOLD calibration

bool birdDetected = false;
float peakMagnitudeInWindow = 0.0f;
unsigned long windowStart = 0;

void setup() {
  pinMode(micPin, INPUT);
  RPC.begin();
  windowStart = millis();
}

float runGoertzelBlock() {
  float s1 = 0.0f, s2 = 0.0f;
  const unsigned long sampleIntervalUs = (unsigned long)(1000000.0f / SAMPLE_RATE_HZ);
  unsigned long nextSampleAt = micros();

  for (int n = 0; n < BLOCK_SIZE; n++) {
    while ((long)(micros() - nextSampleAt) < 0) {
      // wait for the next sample tick
    }
    nextSampleAt += sampleIntervalUs;

    float x = (float)(analogRead(micPin) - ADC_MIDPOINT);
    float s0 = x + COEFF * s1 - s2;
    s2 = s1;
    s1 = s0;
  }

  float real = s1 - s2 * cosf(OMEGA);
  float imag = s2 * sinf(OMEGA);
  return sqrtf(real * real + imag * imag);
}

void loop() {
  float magnitude = runGoertzelBlock();

  if (magnitude > peakMagnitudeInWindow) {
    peakMagnitudeInWindow = magnitude;
  }

  if (magnitude > MAG_THRESHOLD) {
    if (!birdDetected) {
      birdDetected = true;
      RPC.call("handleBird", "BIRD");
    }
  } else {
    birdDetected = false;
  }

  if (millis() - windowStart >= REPORT_INTERVAL_MS) {
    RPC.call("reportMagnitude", (int)peakMagnitudeInWindow);
    peakMagnitudeInWindow = 0.0f;
    windowStart = millis();
  }
}
