#!/usr/bin/python3
import numpy as np
import os
from typing import List
from src.spectra import Spectra
from src.fileManager import FileManager
from src.graph import Graph
from src.parameters import SpectraFragments


def getSpectraList(manager: FileManager):
    spectras = []
    for file in manager.fileList:
        spectra = Spectra()
        spectra.setSpectraName(file[:-4])
        spectra.importDataFromCSV(path + file)
        spectras.append(spectra)
    return spectras


def plotSpectrasWithGallPeaks(spectras : List[Spectra], path: str):
    for spectra in spectras:
        graph = Graph(dpi = 250, spectra = spectra)
        graph.plotGraphWithGallPeaks(path = path)


def getCroppedSpectras(spectras: List[Spectra], limits, suffix: str):
    newSpectras = []
    for spectra in spectras:
        newSpectra = spectra.crop(shiftLimits = limits, suffix = suffix)
        newSpectras.append(newSpectra)

    return newSpectras


def graphSpectras(spectras: List[Spectra], path: str, dirName: str, override: bool, limits: tuple):
    for spectra in spectras:
        spectra.saveSpectra(path = path, dirName = dirName, override = override)
        graph = Graph(dpi = 250, spectra = spectra)
        graph.plotGraph(limits = limits, legend = [], path = path + dirName, override = override)
    return


if __name__ == '__main__':
    folders = ["archival/", "dyneema/", "medit/", "vistula/", "wzorzec/", "wzorzec_miliQ/"]
    dataPath = "data/"
    override = False
    for folder in folders:
        path = os.path.join(dataPath, folder + "csvCombined/")
        fileManager = FileManager(path = path)
        spectraList : List[Spectra] = getSpectraList(manager = fileManager)
        fragments: SpectraFragments = SpectraFragments()
        for index in range(len(fragments.rangeNames)):
            croppedSpectras: List[Spectra] = getCroppedSpectras(spectras = spectraList,
                                                                limits = fragments.rangeLimits[index],
                                                                suffix = fragments.rangeNames[index])

            dirName = fragments.rangeNames[index] + "/"
            graphSpectras(limits = fragments.rangeLimits[index],
                          spectras = croppedSpectras,
                          path = path,
                          dirName = dirName,
                          override = override)

        for spectraIndex in croppedSpectras:
            dataFileName = fragments.rangeNames[index] + ".CSV"
            


        # TO DO:
        # 0) Shift stability (to copy + eventually some corrections)
        # 1) Crystalsy based on pure intensity (to copy)
        # 2) Correcting baseline (to copy)
        #   - asLS
        #   - arPLS
        # 3) Deconvolution (to copy + organise deconvolution parameters)
        #   - Gauss
        #   - Lorentz
        # 4) Check the level of model fit
        # 5) Crystals based on deconvolution (to copy)




            

