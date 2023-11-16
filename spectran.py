#!/usr/bin/python3
import os
# import numpy as np
from typing import List
from src.spectra import Spectra
from src.fileManager import FileManager
from src.parameters import SpectraFragments
from src.manager import Manager

if __name__ == '__main__':
    rootFolder = "data/"
    probeTypeDirs = ["archival/", "dyneema/", "medit/", "vistula/", "wzorzec/", "wzorzec_miliQ/"]
    vibrationTypeDirs = ["bend/", "ccstretch/", "stretch/", "twist/"]
    correctionMethodDirs = ["asLS/", "arLS/"]
    override = False
    for folder in probeTypeDirs:
        path = os.path.join(rootFolder, folder + "csvCombined/")
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
                        folderPath = rootFolder + folder + "rsStability/"
                        newFilePath = folderPath + dataFileName
                        if not os.path.isdir(folderPath) and saveRS:
                            os.mkdir(folderPath)

                        peakStats = spectra.findPeakDifferences(signal = values[1])
                        if saveRS:
                            print("Saving rs stability data to: " + newFilePath)
                            Manager.savePeakStats(filePath = newFilePath, stats = peakStats)
                    else:
                        continue

                spectrasPath = path + dirName

                if not Manager.CheckIfSpectraAsLSCorrected(spectraName = spectra.name, path = spectrasPath, dirName = "asLS/"):
                    Manager.correctAsLS(spectra = spectra, lamb = 3e8, termPrecision=0.32)
                    Manager.saveAsLSCorrection(spectra = spectra, path = spectrasPath, dirName = "asLS/",
                                               override = True, plot = True)
                else:
                    spectra.LoadAsLSCorrection(path = spectrasPath, dirName = "asLS/")

                if not Manager.CheckIfSpectraArLSCorrected(spectraName = spectra.name, path = spectrasPath, dirName = "arLS/"):
                    Manager.correctArLS(spectra = spectra, lam = 3e8, asymWeight = 0.01)
                    Manager.saveArLSCorrection(spectra = spectra, path = spectrasPath, dirName = "arLS/",
                                               override = True, plot = True)
                else:
                    spectra.LoadArLSCorrection(path = spectrasPath, dirName = "arLS/")

                Manager.saveArLSAsLSComparison(spectra = spectra, path = spectrasPath, dirName = "methodComparison/",
                                               override = False, plot = False)

    for probeType in probeTypeDirs:
        path = os.path.join(rootFolder, probeType + "csvCombined/")
        #Manager.calculateRawCrysts(path = path, probeType = probeType.replace("/", ""))
        #Manager.deconv(path = path)
        print("Starting calculating deconv crysts")
        Manager.calculateDeconvCrysts(path = path, probeType = probeType)
        # preparation for deconvolution
        # TO DO:
        # 0) Shift stability (to copy + eventually some corrections) DONE
        # 1) Crystalsy based on pure intensity (to copy)
        # 2) Correcting baseline (to copy) DONE
        #   - asLS DONE
        #   - arPLS DONE
        # 3) Deconvolution (to copy + organise deconvolution parameters) TODO
        #   - Gauss TODO
        #   - Lorentz TODO
        # 4) Check the level of model fit
        # 5) Crystals based on deconvolution (to copy) TODO
