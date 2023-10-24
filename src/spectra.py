import numpy as np
import os
from scipy.signal import find_peaks


class Spectra:
    intensities = []
    shifts = []
    name: str = ""
    
    def __init__(self):
        pass

    def setSpectraName(self, name: str):
        self.name = name

    def setSpectraIntensities(self, intensities):
        self.intensities = intensities

    def setSpectraShifts(self, shifts):
        self.shifts = shifts

    def importDataFromCSV(self, filePath):
        data = np.loadtxt(filePath, delimiter = ',')
        self.shifts = np.array(data[:, 0], dtype = 'float')
        self.intensities = np.array(data[:, 1], dtype = 'float')

    def crop(self, shiftLimits, suffix: str):
        croppedShifts = np.array([])
        croppedIntensities = np.array([])

        for i in range(len(self.shifts)):
            shift = self.shifts[i]
            intensity = self.intensities[i]
            
            if (shift < shiftLimits[0]) or (shift > shiftLimits[1]):
                continue
            croppedShifts = np.append(croppedShifts, shift)
            croppedIntensities = np.append(croppedIntensities, intensity)
            
        newSpectra = Spectra()
        newSpectra.setSpectraName(suffix + "_" + self.name)
        newSpectra.setSpectraIntensities(croppedIntensities)
        newSpectra.setSpectraShifts(croppedShifts)

        return newSpectra

    def saveSpectra(self, path: str, dirName: str, override: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        filePath = dirPath + self.name + ".CSV"
        if os.path.exists(filePath) and (not override):
            print("Passing")
            return
        file = open(dirPath + self.name + ".CSV", "w")
        for index in range(len(self.shifts)):
            file.write(str(self.shifts[index]) + "," + str(self.intensities[index]) + "\n")
        file.close()

    def findPeaks(self, path: str, fileName: str, save: bool):
        peaksIndexes = find_peaks(self.intensities, prominence = 2)[0]
        peaksSpectraShifts = np.array(self.shifts[peaksIndexes], dtype= 'float')
        filteredPeaksShifts = np.array([], dtype = 'float')



        return

