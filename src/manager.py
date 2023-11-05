from src.spectra import Spectra
from src.fileManager import FileManager
from src.logger import Logger
from src.graph import Graph
import numpy as np
from typing import List
import scipy as sc
import os

class Manager:
    def __init__(self):
        pass

    @staticmethod
    def getSpectraList(manager: FileManager) -> np.array:
        spectras = np.array([])
        Logger.logInfo(message = "-----Importing data-----")
        for file in manager.fileList:
            Logger.logInfo("Importing data for " + file)
            spectra = Spectra()
            spectra.setSpectraName(file[:-4])
            spectra.importDataFromCSV(manager.path + file)
            spectras = np.append(spectras, spectra)
            Logger.logInfo(message = "Imported data for " + file)
        Logger.logInfo(message = "-----Finished importing data-----")
        return spectras

    @staticmethod
    def plotSpectrasWithGallPeaks(spectras: List[Spectra], path: str):
        Logger.logInfo(message = "-----Plotting graphs with Gal peaks-----")
        for spectra in spectras:
            graph = Graph(dpi = 250, spectra = spectra)
            graph.plotGraphWithGallPeaks(path = path)
        Logger.logInfo(message = "-----Finished plotting graphs with Gal peaks-----")

    @staticmethod
    def getCroppedSpectras(spectras: List[Spectra], limits: object, suffix: str) -> np.array:
        newSpectras = np.array([])
        Logger.logInfo(message = "-----Starting cropping graphs-----")
        for spectra in spectras:
            newSpectra = spectra.crop(shiftLimits = limits, suffix = suffix)
            newSpectras = np.append(newSpectras, newSpectra)
        Logger.logInfo(message = "-----Finished cropping graphs-----")
        return newSpectras

    @staticmethod
    def graphSpectras(spectras: List[Spectra], path: str, dirName: str, override: bool, limits: tuple):
        Logger.logInfo(message = "-----Starting plotting graphs-----")
        for spectra in spectras:
            spectra.saveSpectra(path = path, dirName = dirName, override = override)
            graph = Graph(dpi = 250, spectra = spectra)
            graph.plotGraph(limits = limits, legend = [], path = path + dirName, override = override)
        Logger.logInfo(message = "-----Finished plotting graphs-----")
        return

    @staticmethod
    def savePeakStats(filePath: str, stats):
        file = open(file = filePath, mode = "a")
        file.write(str(stats[0]) + "," + str(stats[1]) + "," + str(stats[2]) + "\n")
        file.close()
        pass


    @staticmethod
    def correctAsLS(spectra: Spectra, lamb: float, termPrecision: float):
        print("Correcting spectra: " + spectra.name + " with asLS")
        spectraSize = len(spectra.intensities)
        e = np.ones(spectraSize, dtype='float64')
        values = np.array([e, -2 * e, e])
        diags = np.array([0, 1, 2])
        D = sc.sparse.spdiags(values, diags, spectraSize - 2, spectraSize).toarray()
        H = lamb * np.matmul(D.transpose(), D)
        w = np.ones(spectraSize)

        while (True):
            W = sc.sparse.spdiags(w, 0, spectraSize, spectraSize)
            C = sc.linalg.cholesky(W + H)
            estBaseline = sc.linalg.solve(C, sc.linalg.solve(C.transpose(), np.multiply(w, spectra.intensities)))
            d = spectra.intensities - estBaseline
            dNegativeElems = d[d < 0]
            dNegMean = np.mean(dNegativeElems)
            dstdDev = np.std(dNegativeElems)
            wt = np.zeros(spectraSize, dtype='float64')

            for i in range(spectraSize):
                wt[i] = 1. / (1 + np.exp(2 * (d[i] - (2 * dstdDev - dNegMean)) / dstdDev))

            if (np.linalg.norm(w - wt) / np.linalg.norm(w)) < termPrecision:
                break

            w = wt

        newSignal = spectra.intensities - estBaseline
        print("Finished correcting " + spectra.name)
        spectra.asLSIntensities = newSignal

    @staticmethod
    def correctArLS(spectra: Spectra, lam: float, asymWeight: float):
        print("Correcting spectra: " + spectra.name + " with arLS")
        spectraSize = len(spectra.intensities)
        e = np.ones(spectraSize, dtype='float64')
        values = np.array([e, -2 * e, e])
        diags = np.array([0, 1, 2])
        D = sc.sparse.spdiags(values, diags, spectraSize - 2, spectraSize).toarray()
        w = np.ones(spectraSize)
        for i in range(10):
            W = sc.sparse.spdiags(w, 0, spectraSize, spectraSize)
            C = sc.linalg.cholesky(W + lam * np.matmul(D.transpose(), D))
            estBaseline = sc.linalg.solve(C, sc.linalg.solve(C.transpose(), np.multiply(w, spectra.intensities)))
            for i in range(spectraSize):
                if spectra.intensities[i] > estBaseline[i]:
                    w[i] = asymWeight
                else:
                    w[i] = 1 - asymWeight

        newSignal = spectra.intensities - estBaseline
        print("Finished correcting " + spectra.name)

        spectra.arLSIntensities = newSignal

    @staticmethod
    def saveAsLSCorrection(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        filePath = dirPath + "asLS_" + spectra.name + ".CSV"
        if os.path.exists(filePath) and (not override):
            return
        file = open(filePath, "w")
        for index in range(len(spectra.asLSIntensities)):
            file.write(str(spectra.shifts[index]) + "," + str(spectra.asLSIntensities[index]) + "\n")

        if plot:
            graph = Graph(spectra = spectra, dpi = 250)
            graph.plotGraphAsLS(spectra = spectra, path = dirPath)
        file.close()


    @staticmethod
    def saveArLSCorrection(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        filePath = dirPath + "arLS_" + spectra.name + ".CSV"
        if os.path.exists(filePath) and (not override):
            return
        file = open(filePath, "w")
        for index in range(len(spectra.asLSIntensities)):
            file.write(str(spectra.shifts[index]) + "," + str(spectra.asLSIntensities[index]) + "\n")

        if plot:
            graph = Graph(spectra=spectra, dpi=250)
            graph.plotGraphArLS(spectra=spectra, path=dirPath)
        file.close()
