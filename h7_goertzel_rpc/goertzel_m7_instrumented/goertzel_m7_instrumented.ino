// goertzel_m7_instrumented.ino — M7 receiver for goertzel_m4_instrumented.
// M4 has no direct USB Serial, so it RPC-relays timing/magnitude data
// here; this sketch just prints. Same pattern as
// h7_drongonet_m4_instrumented's M7 receiver.

#include "RPC.h"

int reportReady(int numBlocks, int blockSize) {
  Serial.println(F("=== goertzel_m4_instrumented: Goertzel filter on M4 ==="));
  Serial.print(F("[OK]  blocks/window: "));
  Serial.print(numBlocks);
  Serial.print(F("  block size: "));
  Serial.println(blockSize);
  Serial.println(F("[OK]  Ready.\n"));
  return 0;
}

int reportGoertzel(int peakMag, int blocksAboveThreshold, int windowUs) {
  Serial.print(F("[GOERTZEL] peakMag="));
  Serial.print(peakMag);
  Serial.print(F("  blocksAboveThreshold="));
  Serial.print(blocksAboveThreshold);
  Serial.print(F("  window="));
  Serial.print(windowUs);
  Serial.println(F(" us"));
  return 0;
}

int reportGoertzelAverage(int count, int avgWindowUs) {
  Serial.println();
  Serial.println(F("=== Goertzel M4 characterization complete ==="));
  Serial.print(F("[AVG over ")); Serial.print(count); Serial.println(F(" windows]"));
  Serial.print(F("  avg window: ")); Serial.print(avgWindowUs); Serial.println(F(" us"));
  return 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  delay(1000);

  RPC.begin();
  RPC.bind("reportReady",          reportReady);
  RPC.bind("reportGoertzel",       reportGoertzel);
  RPC.bind("reportGoertzelAverage", reportGoertzelAverage);
}

void loop() {
  yield();
}
