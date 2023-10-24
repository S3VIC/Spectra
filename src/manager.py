from src.spectra import Spectra
from src.fileManager import FileManager
from src.logger import Logger
from src.graph import Graph
import numpy as np
from typing import List


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
