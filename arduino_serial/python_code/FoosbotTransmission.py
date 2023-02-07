import threading
import serial
import queue

class FoosbotTransmission:
    def __init__(self, port, baud_rate):
        self.port = port
        self.baud_rate = baud_rate

        self.done = False

        self.serial = serial.Serial(port, baud_rate)
        # clear buffer
        self.serial.reset_input_buffer()
        self.serial.cancel_write()

        self.data_queue = queue.Queue()

        # Start asynchronous read
        self.read_thread = threading.Thread(target=self.asynchronous_read)
        self.read_thread.start()

    def close(self):
        self.serial.close()
        self.done = True
        self.read_thread.join()

    def asynchronous_read(self):
        while not self.done:
            try:
                data = self.serial.readline()
                self.data_queue.put(data)
            except Exception as e:
                print("Exception in asynchronous read: ", e)


    def get_data(self):
        if self.data_queue.empty():
            return None

        data = self.data_queue.get()

        try:
            return data.decode("utf-8").strip()
        except UnicodeDecodeError:
            # Send back an error message
            print("Error decoding data: ", data)
            return None

    def send_data(self, data, encoding="utf-8"):
        if type(data) != bytes:
            data = data.encode(encoding)

        self.serial.write(data)