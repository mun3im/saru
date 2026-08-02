#include "RPC.h"
#include <string>  // Required for std::string

int handleBird(std::string msg) {
  if (!msg.empty()) {
    Serial.println(msg.c_str());  // Prints "BIRD" to the USB serial (tty)
	  digitalWrite(LEDR, HIGH);   // OFF
	  delay(500);
	  digitalWrite(LEDR, LOW);    // ON
  }
  return 0;
}

int reportMagnitude(int mag) {
  Serial.print("Goertzel peak magnitude (last 1s): ");
  Serial.println(mag);
  return 0;
}

void setup() {
  Serial.begin(115200);
  pinMode(LEDR, OUTPUT);      // Red LED for M7
  digitalWrite(LEDR, HIGH);   // Start OFF (Active-LOW)

  RPC.begin();
  RPC.bind("handleBird", handleBird);
  RPC.bind("reportMagnitude", reportMagnitude);
}

void loop() {
  yield();
}
