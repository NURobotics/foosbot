#include <Arduino.h>

#define SERIAL_BAUD 115200

void clearSerial() {
  while (Serial.available()) {
    Serial.read();
  }
}

void setup() {
  // put your setup code here, to run once:
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(SERIAL_BAUD);
  clearSerial();
}

long last = 0;

void loop() {
  last++;

  // check if serial buffer is full, if not send data

  // Create a string with the data
  String data = String(last) + " " + String(millis()) + "\n";

  if (Serial.availableForWrite() > data.length()) {
    Serial.print(data);
  }

  while (Serial.available()) {
    String c = Serial.readString();

    if (c[0] == '1') {
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (c[0] == '0') {
      digitalWrite(LED_BUILTIN, LOW);
    }
  }
  delay(1);
}