#include <Arduino.h>

#define SERIAL_BAUD 115200

struct DataPacket {
    uint32_t counter;
    uint32_t time;
    };

void clearSerial();

void setupSerial();

bool attemptWriteSerial(DataPacket data);

