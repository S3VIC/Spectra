

class Logger:
    def __init__(self):
        pass

    @staticmethod
    def logInfo(message: str):
        print("[INFO]: " + message)

    @staticmethod
    def logWarning(message: str):
        print("[WARNING]: " + message)

    @staticmethod
    def logError(message: str):
        print("[ERROR]: " + message)
