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
    rootOutputDeconvPath = "/home/sewik/source/repos/Spectra/deconv/"
    #probeTypeDirs = ["archival/", "dyneema/", "medit/", "vistula/", "wzorzec/", "wzorzec_miliQ/"]
    #probeTypeDirs = ["wzorzec/", "dyneema/"]
    probeTypeDirs = ["archival/", "medit/", "vistula/", "dyneema/", "wzorzec/"]
    vibrationTypeDirs = ["bend/", "ccstretch/", "stretch/", "twist/"]
    correctionMethodDirs = ["asLS/", "arLS/"]
    override = True
    lambdaRange = [3e5, 3e6, 3e7, 3e8, 3e9]
    asLS_termPrecissionRange = [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
    asymWeightRange = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    #for folder in probeTypeDirs:
    #    path = os.path.join(rootFolder, folder + "csvCombined/")
    #    fileManager = FileManager(path=path)
    #    spectraList: List[Spectra] = Manager.getSpectraList(manager=fileManager)
    #    fragments: SpectraFragments = SpectraFragments()
    #    for index in range(len(fragments.rangeNames)):
    #        croppedSpectras: List[Spectra] = Manager.getCroppedSpectras(spectras=spectraList,
    #                                                                    limits=fragments.rangeLimits[index],
    #                                                                    suffix=fragments.rangeNames[index])
    #        dirName = fragments.rangeNames[index] + "/"
    #        Manager.graphSpectras(limits=fragments.rangeLimits[index],
    #                              spectras=croppedSpectras,
    #                              path=path,
    #                              dirName=dirName,
    #                              override=False)
    #        for spectra in croppedSpectras:
    #            spectra.findPeaks()
    #            saveRS = False
    #            for name, values in fragments.signals.items():
    #                if values[0] == index + 1:
    #                    dataFileName = name + "_rs_stability.CSV"
    #                    folderPath = rootFolder + folder + "rsStability/"
    #                    newFilePath = folderPath + dataFileName
    #                    if not os.path.isdir(folderPath) and saveRS:
    #                        os.mkdir(folderPath)

    #                    peakStats = spectra.findPeakDifferences(signal = values[1])
    #                    if saveRS:
    #                        print("Saving rs stability data to: " + newFilePath)
    #                        Manager.savePeakStats(filePath = newFilePath, stats = peakStats)
    #                else:
    #                    continue

    #            spectrasPath = path + dirName

    #            if not Manager.CheckIfSpectraAsLSCorrected(spectraName = spectra.name, path = spectrasPath, dirName = "asLS/") or override:
    #                for lamb in lambdaRange:
    #                    for termPrecission in asLS_termPrecissionRange:
    #                        print(f"Performing asLS for:\nlamd={lamb}\nterm={termPrecission}\n")
    #                        Manager.correctAsLS(spectra = spectra, lamb = lamb, termPrecision=termPrecission)
    #                        Manager.saveAsLSCorrection(spectra = spectra, path = spectrasPath, dirName = "asLS/",
    #                                                   override = False, plot = True, lamb=str(lamb), term=str(termPrecission))
    #            else:
    #                spectra.LoadAsLSCorrection(path = spectrasPath, dirName = "asLS/")

    #            if not Manager.CheckIfSpectraArLSCorrected(spectraName = spectra.name, path = spectrasPath, dirName = "arLS/") or override:
    #                for lamb in lambdaRange:
    #                    for weight in asymWeightRange:
    #                        print(f"Performing asLS for:\nlamd={lamb}\nweight={weight}\n")
    #                        Manager.correctArLS(spectra = spectra, lam = lamb, asymWeight = weight)
    #                        Manager.saveArLSCorrection(spectra = spectra, path = spectrasPath, dirName = "arLS/",
    #                                               override = False, plot = True, lamb=str(lamb), weight=str(weight))
    #            else:
    #                spectra.LoadArLSCorrection(path = spectrasPath, dirName = "arLS/")

                #Manager.saveArLSAsLSComparison(spectra = spectra, path = spectrasPath, dirName = "methodComparison/",
                #                               override = False, plot = False)

    #for probeType in probeTypeDirs:
    #    for method in correctionMethodDirs:
    #        path = os.path.join(rootFolder, probeType + "csvCombined/")
    #        if method == 'asLS/':
    #            param2Values = asLS_termPrecissionRange
    #        else:
    #            param2Values = asymWeightRange
    #        for lamb in lambdaRange:
    #            for param2 in param2Values:
    #                Manager.calculateRawCrysts(
    #                    path = path,
    #                    probeType = probeType.replace("/", ""),
    #                    param1 = str(lamb),
    #                    param2 = str(param2),
    #                    pathToSave=method + probeType + str(lamb),
    #                    method = method
    #                )
    for probeType in probeTypeDirs:
        for method in correctionMethodDirs:
            path = os.path.join(rootFolder, probeType, "csvCombined")
            Manager.deconv(path = path, lambdaRange = lambdaRange, asLSSecond = asLS_termPrecissionRange, arLSSecond = asymWeightRange, rootOutputPath = rootOutputDeconvPath)
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
