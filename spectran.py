#!/usr/bin/python3
import os
import numpy as np
from typing import List
from src.spectra import Spectra
from src.fileManager import FileManager
from src.parameters import SpectraFragments
from src.manager import Manager

if __name__ == '__main__':
    folders = ["archival/", "dyneema/", "medit/", "vistula/", "wzorzec/", "wzorzec_miliQ/"]
    dataPath = "data/"
    override = False
    for folder in folders:
        path = os.path.join(dataPath, folder + "csvCombined/")
        fileManager = FileManager(path=path)
        spectraList: List[Spectra] = Manager.getSpectraList(manager=fileManager)
        fragments: SpectraFragments = SpectraFragments()
        for index in range(len(fragments.rangeNames)):
            croppedSpectras: List[Spectra] = Manager.getCroppedSpectras(spectras=spectraList,
                                                                        limits=fragments.rangeLimits[index],
                                                                        suffix=fragments.rangeNames[index])
            dirName = fragments.rangeNames[index] + "/"
            Manager.graphSpectras(limits=fragments.rangeLimits[index],
                                  spectras=croppedSpectras,
                                  path=path,
                                  dirName=dirName,
                                  override=override)
            for spectra in croppedSpectras:
                spectra.findPeaks()
                saveRS = False
                for name, values in fragments.signals.items():
                    if values[0] == index + 1:
                        dataFileName = name + "_rs_stability.CSV"
                        folderPath = dataPath + folder + "rsStability/"
                        newFilePath = folderPath + dataFileName
                        if not os.path.isdir(folderPath) and saveRS:
                            os.mkdir(folderPath)

                        peakStats = spectra.findPeakDifferences(signal = values[1])
                        #print("Found diff: " + str(peakStats[0]) + " with position: " + str(peakStats[1]) +
                        #      " for reference: " + str(values[1]))
                        if saveRS:
                            print("Saving rs stability data to: " + newFilePath)
                            Manager.savePeakStats(filePath = newFilePath, stats = peakStats)
                    else:
                        continue

                Manager.correctAsLS(spectra = spectra, lamb = 1e8, termPrecision=0.25)
                spectrasPath = path + dirName
                Manager.saveAsLSCorrection(spectra = spectra, path = spectrasPath, dirName = "asLS/", override = False,
                                           plot = False)
                Manager.correctArLS(spectra = spectra, lam = 1e8, asymWeight = 0.01)
                Manager.saveArLSCorrection(spectra = spectra, path = spectrasPath, dirName = "arLS/", override = True,
                                           plot = True)

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
