// Analog Microphone Test with LED Sound Detection Indicator
// MAX4466 connected to A1
// Uses built-in RGB LEDs for visual feedback

const int microphonePin = A1;

// Sound detection threshold - adjust based on your environment
const int SOUND_THRESHOLD = 1000;  // Delta threshold for sound detection

void setup() {
  Serial.begin(115200);
  
  // Initialize built-in RGB LEDs as outputs
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  
  // Turn off all LEDs initially (LEDs are active LOW on Portenta H7)
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  
  Serial.println("=== Analog Mic Test with LED Indicator ===");
  Serial.println("Green LED: Sound detected");
  Serial.println("Blue LED: Idle/No sound");
}

void loop() {
  int mn = 1024;
  int mx = 0;

  // Take 10000 samples to capture audio amplitude
  for (int i = 0; i < 10000; i++) {
    int val = analogRead(microphonePin);
    mn = min(mn, val);
    mx = max(mx, val);
  }

  int delta = mx - mn;  // Signal amplitude

  // Print values to Serial Monitor/Plotter
  Serial.print("Min = ");
  Serial.print(mn);
  Serial.print("\tMax = ");
  Serial.print(mx);
  Serial.print("\tDelta = ");
  Serial.print(delta);
  
  // LED indication based on sound detection
  if (delta > SOUND_THRESHOLD) {
    // Sound detected - Turn on GREEN LED, turn off others
    digitalWrite(LEDG, LOW);   // GREEN ON (active LOW)
    digitalWrite(LEDB, HIGH);  // BLUE OFF
    digitalWrite(LEDR, HIGH);  // RED OFF
    Serial.println("\t[SOUND DETECTED]");
  } else {
    // No significant sound - Turn on BLUE LED (idle state)
    digitalWrite(LEDB, LOW);   // BLUE ON (active LOW)
    digitalWrite(LEDG, HIGH);  // GREEN OFF
    digitalWrite(LEDR, HIGH);  // RED OFF
    Serial.println("\t[Idle]");
  }
}