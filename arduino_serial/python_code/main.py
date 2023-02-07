import serial
import serial.tools.list_ports

import time
from FoosbotTransmission import FoosbotTransmission

BAUD_RATE = 115200

def select_com_port(force=False):
    # Get list of serial ports
    ports = serial.tools.list_ports.comports()

    arduino_ports = []

    for i, v in enumerate(ports):
        port, desc, hwid = v

        print(f"{i}: {port} - {desc} - {hwid}", end="")
        if "USB" in desc:
            arduino_ports.append((port, desc, hwid))

            print("  <-- USB port", end="")

        print()

    if len(arduino_ports) > 0 and force:
        return arduino_ports[0]

    user_input = input("Select COM port: ")

    try:
        user_input = int(user_input)
    except ValueError:
        print("Invalid input")
        exit(0)

    return ports[user_input]


def main():
    arduino_port = select_com_port()

    foosbot_transmitter = FoosbotTransmission(arduino_port[0], BAUD_RATE)

    while True:
        try:
            # send "1" to arduino
            foosbot_transmitter.send_data("1")
            print("Sent 1")

            time.sleep(1)

            foosbot_transmitter.send_data("0")
            print("Sent 0")

            time.sleep(1)

            data = foosbot_transmitter.get_data()

            start = data
            last = start

            while data := foosbot_transmitter.get_data():
                if data:
                    last = data
                pass

            if start and last:
                start = start.split()
                last = last.split()
                print(start)
                print(last)

                elapsed_0 = int(last[0]) - int(start[0])
                elapsed_1 = int(last[1]) - int(start[1])

                print(f"Elapsed values: {elapsed_0}, {elapsed_1}")

                print()

        except KeyboardInterrupt:
            foosbot_transmitter.close()
            break

if __name__ == "__main__":
    main()
