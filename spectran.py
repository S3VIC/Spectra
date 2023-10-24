#!/usr/bin/python3
import numpy as np
import os
from typing import List
from src.spectra import Spectra
from src.fileManager import FileManager
from src.graph import Graph
from src.parameters import SpectraFragments

def getSpectraList(fileManager: FileManager):
    spectraList = np.array([])
    for file in fileManager.fileList:
        spectra = Spectra()
        spectra.setSpectraName(file[:-4])
        spectra.importDataFromCSV(path + file)
        spectraList = np.append(spectraList, spectra)
    return spectraList


def plotSpectrasWithGallPeaks(spectraList: List[Spectra], path: str):
    for spectra in spectraList:
        graph = Graph(spectra.shifts, spectra.intensities)
        graph.plotGraphWithGallPeaks(path = path, spectraName = spectra.name)


def getCroppedSpectras(spectraList: List[Spectra], limits, suffix: str):
    newSpectraList = np.array([])
    for spectra in spectraList:
        newSpectra = spectra.crop(shiftLimits = limits, suffix = suffix)
        newSpectraList = np.append(newSpectraList, newSpectra)

    return newSpectraList

def graphSpectras(spectras: List[Spectra], path: str, dirName: str):
    return

if __name__ == '__main__':
    folders = ["archival/", "dyneema/", "medit/", "vistula/", "wzorzec/", "wzorzec_miliQ/"]
    dataPath = "data/"
    override = False
    for folder in folders:
        path = os.path.join(dataPath, folder + "csvCombined/")
        fileManager = FileManager(path = path)
        spectraList : List[Spectra] = getSpectraList(fileManager = fileManager)
        fragments: SpectraFragments = SpectraFragments()
        for index in range(len(fragments.rangeNames)):
            croppedSpectras: List[Spectra] = getCroppedSpectras(spectraList = spectraList,
                                                                limits = fragments.rangeLimits[index],
                                                                suffix = fragments.rangeNames[index])
            for spectraIndex in range(len(croppedSpectras)):
                dirName = fragments.rangeNames[index] + "/"
                croppedSpectras[spectraIndex].saveSpectra(path = path, dirName = dirName, override = override)
                graph = Graph(dpi = 250)
                graph.plotGraph(limits = fragments.rangeLimits[index], legend = [], spectra = croppedSpectras[spectraIndex], path = path + dirName, override = override)
        
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




            

