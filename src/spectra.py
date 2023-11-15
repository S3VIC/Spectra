import numpy as np
import os
from scipy.signal import find_peaks

class Spectra:
    intensities = []
    shifts = []
    name: str = ""
    peakShifts = []
    asLSIntensities = []
    arLSIntensities = []

    def __init__(self):
        intensities = []
        shifts = []
        name: str = ""
        peakShifts = []
        asLSIntensities = []
        arLSIntensities = []
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
        newSpectra.setSpectraName(self.name)
        newSpectra.setSpectraIntensities(croppedIntensities)
        newSpectra.setSpectraShifts(croppedShifts)

        return newSpectra

    def saveSpectra(self, path: str, dirName: str, override: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        filePath = dirPath + self.name + ".CSV"
        if os.path.exists(filePath) and (not override):
            return
        file = open(filePath, "w")
        for index in range(len(self.shifts)):
            file.write(str(self.shifts[index]) + "," + str(self.intensities[index]) + "\n")
        file.close()

    def findPeaks(self):
        peaksIndexes = find_peaks(self.intensities, prominence = 2)[0]
        self.peakShifts = np.array(self.shifts[peaksIndexes], dtype= 'float')

    def findPeakDifferences(self, signal: int):
        if len(self.peakShifts) == 0:
            return -1, -1, -1
        diff = signal - self.peakShifts[0]
        peakPosition = self.peakShifts[0]
        for shiftIndex in range(1, len(self.peakShifts)):
            newDiff = signal - self.peakShifts[shiftIndex]
            if abs(newDiff) <= abs(diff):
                diff = newDiff
                peakPosition = self.peakShifts[shiftIndex]
            else:
                continue

        return diff, peakPosition, signal

    def LoadArLSCorrection(self, path: str, dirName: str):
        dirPath = os.path.join(path, dirName)
        fileName = self.name + ".CSV"
        filePath = dirPath + fileName
        data = np.loadtxt(fname=filePath, delimiter=",")
        self.arLSIntensities = np.array(data[:, 1], dtype='float')

    def LoadAsLSCorrection(self, path: str, dirName: str):
        dirPath = os.path.join(path, dirName)
        fileName = self.name + ".CSV"
        filePath = dirPath + fileName
        data = np.loadtxt(fname=filePath, delimiter=",")
        self.asLSIntensities = np.array(data[:, 1], dtype='float')
