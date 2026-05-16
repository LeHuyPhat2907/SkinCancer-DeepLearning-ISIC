import sys

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.file = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()