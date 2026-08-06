#include "RPC.h"
#include <string>  // Required for std::string

const int m7Led = LEDR;  // Red LED (Active-LOW)

int handleBird(std::string msg) {
  if (!msg.empty()) {
    Serial.println(msg.c_str());  // Prints "BIRD" to the USB serial (tty)
    digitalWrite(m7Led, LOW);     // ON
    delay(500);
    digitalWrite(m7Led, HIGH);    // OFF
  }
  return 0;
}

int reportScore(int scoreX1000) {
  Serial.print("DrongoNet-nano P(bird) avg (x1000): ");
  Serial.println(scoreX1000);
  return 0;
}

void setup() {
  Serial.begin(115200);
  pinMode(m7Led, OUTPUT);
  digitalWrite(m7Led, HIGH);  // Start OFF (Active-LOW)

  RPC.begin();
  RPC.bind("handleBird", handleBird);
  RPC.bind("reportScore", reportScore);
}

void loop() {
  yield();
}
