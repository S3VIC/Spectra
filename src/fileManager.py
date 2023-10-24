import numpy as np
import os

class FileManager:
    fileList = np.array([])
    path = ""
    def __init__(self, path: str):
        self.fileList = np.array([])
        self.path = path
        for fileName in os.listdir(path):
            if fileName[-3:].upper() != "CSV":
                continue

            temp_file = os.path.join(path, fileName)
            if not os.path.isfile(temp_file):
                return

            self.fileList = np.append(self.fileList, fileName)
