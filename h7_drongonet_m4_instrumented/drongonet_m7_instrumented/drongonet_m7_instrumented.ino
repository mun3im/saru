// drongonet_m7_instrumented.ino — M7 receiver for drongonet_m4_instrumented
// M4 has no direct USB Serial, so it RPC-relays timing/score data here;
// this sketch just prints and (if INA219 is wired) computes energy from
// the received latency figures combined with its own live power reading.
// Same energy formula as h7_drongonet_m7_instrumented / Guide_02:
// Energy (uJ) = Power (mW) x Time (us) / 1000.

#include "RPC.h"
#include <string>

#include <Adafruit_INA219.h>
#include <Wire.h>

Adafruit_INA219 ina219;
bool ina219_ok = false;

int reportReady(int nanoArenaUsed, int microArenaUsed) {
  Serial.println(F("=== drongonet_m4_instrumented: Nano + Micro on M4 ==="));
  Serial.print(F("[OK]  M4 Nano  arena used: "));
  Serial.println(nanoArenaUsed);
  Serial.print(F("[OK]  M4 Micro arena used: "));
  Serial.println(microArenaUsed);
  Serial.println(F("[OK]  Ready.\n"));
  return 0;
}

int reportGate(int rms) {
  Serial.print(F("[GATE] Silent window -- RMS="));
  Serial.print(rms);
  Serial.println(F(" -- inference skipped."));
  return 0;
}

int reportError(std::string msg) {
  Serial.print(F("[ERR] "));
  Serial.println(msg.c_str());
  return 0;
}

// M4 sends fixed-point ints, not floats -- (int)(prob*1000), whole-us
// timings -- same compact-payload convention as h7_goertzel_rpc's
// reportMagnitude / h7_drongonet's reportScore. Decode back to real units
// here on the M7 side, where display precision actually matters.
int reportNano(int rms, int probX1000, int melUs, int inferUs) {
  float probBird = probX1000 / 1000.0f;
  int totalUs = melUs + inferUs;
  Serial.print(F("RMS: "));
  Serial.println(rms);
  Serial.print(F("[NANO]  P(bird)="));
  Serial.print(probBird, 3);
  Serial.print(F("  mel="));
  Serial.print(melUs);
  Serial.print(F(" us  infer="));
  Serial.print(inferUs);
  Serial.print(F(" us  total="));
  Serial.print(totalUs);
  Serial.println(F(" us"));

  if (ina219_ok) {
    float power_mW = ina219.getPower_mW();
    float eMel   = power_mW * ((float)melUs   / 1000.0f);
    float eInfer = power_mW * ((float)inferUs / 1000.0f);
    Serial.print(F("        energy: power="));
    Serial.print(power_mW, 1);
    Serial.print(F(" mW  mel="));
    Serial.print(eMel, 1);
    Serial.print(F(" uJ  infer="));
    Serial.print(eInfer, 1);
    Serial.print(F(" uJ  total="));
    Serial.print(eMel + eInfer, 1);
    Serial.println(F(" uJ"));
  }
  return 0;
}

int reportMicro(int rms, int probX1000, int melUs, int inferUs) {
  float probBird = probX1000 / 1000.0f;
  int totalUs = melUs + inferUs;
  Serial.print(F("[MICRO] P(bird)="));
  Serial.print(probBird, 3);
  Serial.print(F("  mel="));
  Serial.print(melUs);
  Serial.print(F(" us (shared w/ nano)  infer="));
  Serial.print(inferUs);
  Serial.print(F(" us  total="));
  Serial.print(totalUs);
  Serial.println(F(" us"));

  if (ina219_ok) {
    float power_mW = ina219.getPower_mW();
    float eMel   = power_mW * ((float)melUs   / 1000.0f);
    float eInfer = power_mW * ((float)inferUs / 1000.0f);
    Serial.print(F("        energy: power="));
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
  return 0;
}

// Shadow comparison between the trusted f32 mel path and the new Q15
// fixed-point path, on the same real captured clip. maxAbsDiff/
// meanAbsDiff are over the final normalized [0,1] mel features (post
// log + per-clip min-max), sent as (int)(value*1000).
int reportQ15Compare(int rms, int melUsF32, int melUsQ15, int maxAbsDiffX1000, int meanAbsDiffX1000) {
  float speedup = (melUsQ15 > 0) ? (float)melUsF32 / (float)melUsQ15 : 0.0f;
  Serial.print(F("[Q15 CMP] RMS="));
  Serial.print(rms);
  Serial.print(F("  mel_f32="));
  Serial.print(melUsF32);
  Serial.print(F(" us  mel_q15="));
  Serial.print(melUsQ15);
  Serial.print(F(" us  speedup="));
  Serial.print(speedup, 2);
  Serial.print(F("x  maxAbsDiff="));
  Serial.print(maxAbsDiffX1000 / 1000.0f, 3);
  Serial.print(F("  meanAbsDiff="));
  Serial.println(meanAbsDiffX1000 / 1000.0f, 4);
  return 0;
}

int reportMicroAverages(int count, int avgMelUs, int avgInferUs, int avgTotalUs) {
  Serial.println();
  Serial.println(F("=== DrongoNet-Micro M4 characterization complete ==="));
  Serial.print(F("[AVG over ")); Serial.print(count); Serial.println(F(" inferences]"));
  Serial.print(F("  avg mel:   ")); Serial.print(avgMelUs);   Serial.println(F(" us"));
  Serial.print(F("  avg infer: ")); Serial.print(avgInferUs); Serial.println(F(" us"));
  Serial.print(F("  avg total: ")); Serial.print(avgTotalUs); Serial.println(F(" us"));
  return 0;
}

int handleBird(std::string msg) {
  if (!msg.empty()) {
    Serial.println(F("BIRD (nano)"));
    digitalWrite(LEDR, LOW);   // ON
    delay(500);
    digitalWrite(LEDR, HIGH);  // OFF
  }
  return 0;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  delay(1000);

  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, HIGH);  // Start OFF (Active-LOW)

  Wire.begin();
  if (ina219.begin()) {
    ina219_ok = true;
    Serial.println(F("[OK]  INA219 found -- energy will be reported."));
  } else {
    Serial.println(F("[WARN] INA219 not found -- energy reporting skipped."));
  }

  RPC.begin();
  RPC.bind("reportReady", reportReady);
  RPC.bind("reportGate",  reportGate);
  RPC.bind("reportError", reportError);
  RPC.bind("reportNano",  reportNano);
  RPC.bind("reportMicro", reportMicro);
  RPC.bind("reportMicroAverages", reportMicroAverages);
  RPC.bind("reportQ15Compare", reportQ15Compare);
  RPC.bind("handleBird",  handleBird);
}

void loop() {
  yield();
}
