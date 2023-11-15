from src.spectra import Spectra
from src.fileManager import FileManager
from src.logger import Logger
from src.graph import Graph
import numpy as np
from typing import List
import scipy as sc
from scipy.optimize import curve_fit
import os
from src.parameters import GaussParams, LorentzParams
import src.modelFunctions as fn
from src.parameters import DiagnosticSignals

class Manager:
    def __init__(self):
        pass

    @staticmethod
    def getSpectraList(manager: FileManager) -> np.array:
        spectras = np.array([])
        for file in manager.fileList:
            spectra = Spectra()
            spectra.setSpectraName(file[:-4])
            spectra.importDataFromCSV(manager.path + file)
            spectras = np.append(spectras, spectra)
        return spectras

    @staticmethod
    def plotSpectrasWithGallPeaks(spectras: List[Spectra], path: str):
        for spectra in spectras:
            graph = Graph(dpi=250, spectra=spectra)
            graph.plotGraphWithGallPeaks(path=path)

    @staticmethod
    def getCroppedSpectras(spectras: List[Spectra], limits: object, suffix: str) -> np.array:
        newSpectras = np.array([])
        for spectra in spectras:
            newSpectra = spectra.crop(shiftLimits=limits, suffix=suffix)
            newSpectras = np.append(newSpectras, newSpectra)
        return newSpectras

    @staticmethod
    def graphSpectras(spectras: List[Spectra], path: str, dirName: str, override: bool, limits: tuple):
        for spectra in spectras:
            spectra.saveSpectra(path=path, dirName=dirName, override=override)
            graph = Graph(dpi=250, spectra=spectra)
            graph.plotGraph(limits=limits, legend=[], path=path + dirName, override=override)
        return

    @staticmethod
    def savePeakStats(filePath: str, stats):
        file = open(file=filePath, mode="a")
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
        spectra.arLSIntensities = newSignal

    @staticmethod
    def saveAsLSCorrection(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        filePath = dirPath + spectra.name + ".CSV"
        if os.path.exists(filePath) and (not override):
            return
        file = open(filePath, "w")
        for index in range(len(spectra.asLSIntensities)):
            file.write(str(spectra.shifts[index]) + "," + str(spectra.asLSIntensities[index]) + "\n")

        if plot:
            graph = Graph(spectra=spectra, dpi=250)
            graph.plotGraphAsLS(spectra=spectra, path=dirPath)
        file.close()

    @staticmethod
    def saveArLSCorrection(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        filePath = dirPath + spectra.name + ".CSV"
        if os.path.exists(filePath) and (not override):
            return
        file = open(filePath, "w")
        for index in range(len(spectra.asLSIntensities)):
            file.write(str(spectra.shifts[index]) + "," + str(spectra.arLSIntensities[index]) + "\n")

        if plot:
            graph = Graph(spectra=spectra, dpi=250)
            graph.plotGraphArLS(spectra=spectra, path=dirPath)
        file.close()

    @staticmethod
    def saveArLSAsLSComparison(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)
        if plot:
            graph = Graph(spectra=spectra, dpi=250)
            graph.plotGraphAsLSArLSCombined(spectra=spectra, path=dirPath)
        pass

    @staticmethod
    def CheckIfSpectraAsLSCorrected(spectraName: str, path: str, dirName: str):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            return False
        if os.path.exists(dirPath + spectraName + ".CSV"):
            return True
        else:
            return False

    @staticmethod
    def CheckIfSpectraArLSCorrected(spectraName: str, path: str, dirName: str):
        dirPath = os.path.join(path, dirName)
        if not os.path.exists(dirPath):
            return False
        if os.path.exists(dirPath + spectraName + ".CSV"):
            return True
        else:
            return False

    @staticmethod
    def LoadAsLSCorrection(spectra: Spectra, path: str, dirName: str):
        dirPath = os.path.join(path, dirName)
        fileName = spectra.name + ".CSV"
        filePath = dirPath + fileName
        data = np.loadtxt(fname=filePath, delimiter=",")
        spectra.asLSIntensities = np.array(data[:, 1], dtype='float')

    @staticmethod
    def LoadArLSCorrection(spectra: Spectra, path: str, dirName: str):
        dirPath = os.path.join(path, dirName)
        fileName = spectra.name + ".CSV"
        filePath = dirPath + fileName
        data = np.loadtxt(fname=filePath, delimiter=",")
        spectra.asLSIntensities = np.array(data[:, 1], dtype='float')

    @staticmethod
    def deconv(path: str):
        Manager.deconvStretch(path=path, correctionMethod="asLS")
        Manager.deconvStretch(path=path, correctionMethod="arLS")
        Manager.deconvBend(path=path, correctionMethod="asLS")
        Manager.deconvBend(path=path, correctionMethod="arLS")
        Manager.deconvTwist(path=path, correctionMethod="asLS")
        Manager.deconvTwist(path=path, correctionMethod="arLS")
        Manager.deconvCcstretch(path=path, correctionMethod="asLS")
        Manager.deconvCcstretch(path=path, correctionMethod="arLS")

    @staticmethod
    def deconvStretch(path: str, correctionMethod: str):
        vibrationDir = "stretch/"
        spectrasPath = os.path.join(path, vibrationDir + correctionMethod + "/")
        print("Started deconvolution for " + spectrasPath)
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss4term
        funcLorentz = fn.lorentz4term
        for spectra in spectraList:
            if os.path.exists(spectrasPath + spectra.name + ".CSV") and os.path.exists(spectrasPath + spectra.name + ".CSV"):
                continue
            try:
                poptGauss, _ = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                         p0=gaussParams.c1_inits, bounds=gaussParams.c1_bounds)
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3], poptGauss[4],
                                     poptGauss[5], poptGauss[6], poptGauss[7], poptGauss[8], poptGauss[9], poptGauss[10],
                                     poptGauss[11])

                poptLorentz, _ = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                           p0=lorentzParams.c1_inits, bounds=lorentzParams.c1_bounds)
                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5], poptLorentz[6], poptLorentz[7], poptLorentz[8],
                                         poptLorentz[9], poptLorentz[10], poptLorentz[11])
            except RuntimeError:
                print("Optimal parameters not found for " + spectra.name + ", continuing")
                continue

            graph = Graph(spectra=spectra, dpi=250)
            graph.plotDeconvFit(spectra=spectra, fit=fitGauss, path=spectrasPath + "gauss/", override=False)
            graph.plotDeconvFit(spectra=spectra, fit=fitLorentz, path=spectrasPath + "lorentz/", override=False)

        print("Finished deconvolution for " + spectrasPath)



    @staticmethod
    def deconvBend(path: str, correctionMethod: str):
        vibrationDir = "bend/"
        spectrasPath = os.path.join(path, vibrationDir + correctionMethod + "/")
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss4term
        funcLorentz = fn.lorentz4term
        print("Performing bend deconvolution")
        for spectra in spectraList:
            if os.path.exists(spectrasPath + spectra.name + ".CSV") and os.path.exists(spectrasPath + spectra.name + ".CSV"):
                continue
            try:
                poptGauss, _ = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                     p0=gaussParams.c2_inits, bounds=gaussParams.c2_bounds)
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3], poptGauss[4],
                                     poptGauss[5], poptGauss[6], poptGauss[7], poptGauss[8], poptGauss[9], poptGauss[10],
                                     poptGauss[11])

                poptLorentz, _ = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                           p0=lorentzParams.c2_inits, bounds=lorentzParams.c2_bounds)
                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5], poptLorentz[6], poptLorentz[7], poptLorentz[8],
                                         poptLorentz[9], poptLorentz[10], poptLorentz[11])
            except RuntimeError:
                print("Optimal parameters not found, continuing")
                continue

            graph = Graph(spectra=spectra, dpi=250)
            graph.plotDeconvFit(spectra=spectra, fit=fitGauss, path=spectrasPath + "gauss/", override=False)
            graph.plotDeconvFit(spectra=spectra, fit=fitLorentz, path=spectrasPath + "lorentz/", override=False)

    @staticmethod
    def deconvTwist(path: str, correctionMethod: str):
        vibrationDir = "twist/"
        spectrasPath = os.path.join(path, vibrationDir + correctionMethod + "/")
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss2term
        funcLorentz = fn.lorentz2term
        print("Performing ccstretch deconvolution")
        for spectra in spectraList:
            if os.path.exists(spectrasPath + spectra.name + ".CSV") and os.path.exists(spectrasPath + spectra.name + ".CSV"):
                continue
            try:
                poptGauss, _ = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                         p0=gaussParams.c3_inits, bounds=gaussParams.c3_bounds)
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3], poptGauss[4],
                                     poptGauss[5])

                poptLorentz, _ = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                           p0=lorentzParams.c3_inits, bounds=lorentzParams.c3_bounds)
                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5])
            except RuntimeError:
                print("Optimal parameters not found, continuing")
                continue
            graph = Graph(spectra=spectra, dpi=250)
            graph.plotDeconvFit(spectra=spectra, fit=fitGauss, path=spectrasPath + "gauss/", override=False)
            graph.plotDeconvFit(spectra=spectra, fit=fitLorentz, path=spectrasPath + "lorentz/", override=False)

    @staticmethod
    def deconvCcstretch(path: str, correctionMethod: str):
        vibrationDir = "ccstretch/"
        spectrasPath = os.path.join(path, vibrationDir + correctionMethod + "/")
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss2term
        funcLorentz = fn.lorentz2term
        print("Performing ccstretch deconvolution")
        for spectra in spectraList:
            if os.path.exists(spectrasPath + spectra.name + ".CSV") and os.path.exists(spectrasPath + spectra.name + ".CSV"):
                continue
            try:
                poptGauss, _ = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                         p0=gaussParams.c4_inits, bounds=gaussParams.c4_bounds)
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3], poptGauss[4],
                                     poptGauss[5])

                poptLorentz, _ = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                           p0=lorentzParams.c4_inits, bounds=lorentzParams.c4_bounds)
                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5])
            except RuntimeError:
                print("Optimal parameters not found, continuing")
                continue

            graph = Graph(spectra=spectra, dpi=250)
            graph.plotDeconvFit(spectra=spectra, fit=fitGauss, path=spectrasPath + "gauss/", override=False)
            graph.plotDeconvFit(spectra=spectra, fit=fitLorentz, path=spectrasPath + "lorentz/", override=False)


    @staticmethod
    def calculateRawCryst1(path: str, probeType: str):
        fileName = "cryst1-" + probeType + "-raw.CSV"
        if os.path.exists(fileName):
            return
        spectrasDir = "stretch/"
        spectrasPath = os.path.join(path, spectrasDir)
        fileManager = FileManager(spectrasPath)
        spectraList = Manager.getSpectraList(fileManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_str_sym'], SignalsDict['CH3_str_asym']])
        file = open(fileName, "w")
        for spectra in spectraList:
            intensitiesForCryst = np.array([])
            spectra.findPeaks()
            for signal in signals:
                peakPosition = spectra.findPeakDifferences(signal=signal)[1]
                intensitiesForCryst = np.append(intensitiesForCryst, spectra.intensities[np.where(spectra.shifts == peakPosition)])
            cryst = intensitiesForCryst[0] / intensitiesForCryst[1]
            file.write(str(cryst)+'\n')
        file.close()
    @staticmethod
    def calculateRawCryst2(path: str, probeType: str):
        fileName = "cryst2-" + probeType + "-raw.CSV"
        if os.path.exists(fileName):
            return
        spectrasDir = "bend/"
        spectrasPath = os.path.join(path, spectrasDir)
        fileManager = FileManager(spectrasPath)
        spectraList = Manager.getSpectraList(fileManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_ben_cryst'], SignalsDict['CH2_ben_amorf']])
        file = open(fileName, "w")
        for spectra in spectraList:
            intensitiesForCryst = np.array([])
            spectra.findPeaks()
            for signal in signals:
                peakPosition = spectra.findPeakDifferences(signal=signal)[1]
                intensitiesForCryst = np.append(intensitiesForCryst, spectra.intensities[np.where(spectra.shifts == peakPosition)])
            cryst = intensitiesForCryst[0] / intensitiesForCryst[1]
            file.write(str(cryst)+'\n')
        file.close()

    @staticmethod
    def calculateRawCryst3(path: str, probeType: str):
        fileName = "cryst3-" + probeType + "-raw.CSV"
        #if os.path.exists(fileName):
        #    return
        spectrasDir1 = "bend/"
        spectrasDir2 = "twist/"
        spectrasPath1 = os.path.join(path, spectrasDir1)
        spectrasPath2 = os.path.join(path, spectrasDir2)
        bendSpectrasManager = FileManager(spectrasPath1)
        twistSpectrasManager = FileManager(spectrasPath2)
        bendSpectraList = Manager.getSpectraList(bendSpectrasManager)
        twistSpectraList = Manager.getSpectraList(twistSpectrasManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_ben_cryst'], SignalsDict['CH2_twist_amorf']])
        file = open(fileName, "w")
        for spectraTwist in twistSpectraList:
            for spectraBend in bendSpectraList:
                if spectraBend.name == spectraTwist.name:
                    intensitiesForCryst = np.array([])
                    spectraBend.findPeaks()
                    spectraTwist.findPeaks()
                    bendPeakPosition = spectraBend.findPeakDifferences(signal=signals[0])[1]
                    twistPeakPosition = spectraTwist.findPeakDifferences(signal=signals[1])[1]
                    bendIntensity = spectraBend.intensities[np.where(spectraBend.shifts == bendPeakPosition)]
                    twistIntensity = spectraTwist.intensities[np.where(spectraTwist.shifts == twistPeakPosition)]
                    cryst = bendIntensity / twistIntensity
                    file.write(str(cryst).replace("[", "").replace("]", "") +'\n')
        file.close()

    @staticmethod
    def calculateRawCryst4(path: str, probeType: str):
        fileName = "cryst4-" + probeType + "-raw.CSV"
        #if os.path.exists(fileName):
        #    return
        spectrasDir1 = "bend/"
        spectrasDir2 = "ccstretch/"
        spectrasPath1 = os.path.join(path, spectrasDir1)
        spectrasPath2 = os.path.join(path, spectrasDir2)
        bendSpectrasManager = FileManager(spectrasPath1)
        ccstretchSpectrasManager = FileManager(spectrasPath2)
        bendSpectraList = Manager.getSpectraList(bendSpectrasManager)
        ccstretchSpectraList = Manager.getSpectraList(ccstretchSpectrasManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_ben_cryst'], SignalsDict['CC_str_amorf']])
        file = open(fileName, "w")
        for spectraCcstretch in ccstretchSpectraList:
            for spectraBend in bendSpectraList:
                if spectraBend.name == spectraCcstretch.name:
                    spectraBend.findPeaks()
                    spectraCcstretch.findPeaks()
                    bendPeakPosition = spectraBend.findPeakDifferences(signal=signals[0])[1]
                    ccStretchPeakPosition = spectraCcstretch.findPeakDifferences(signal=signals[1])[1]
                    bendIntensity = spectraBend.intensities[np.where(spectraBend.shifts == bendPeakPosition)]
                    ccStretchIntensity = spectraCcstretch.intensities[np.where(spectraCcstretch.shifts == ccStretchPeakPosition)]
                    cryst = bendIntensity / ccStretchIntensity
                    print(type(cryst))
                    file.write(str(cryst).replace("[", "").replace("]", "") +'\n')
        file.close()

    @staticmethod
    def calculateRawCrysts(path: str, probeType: str):
        print("Calculating raw cryst 1 for " + probeType)
        Manager.calculateRawCryst1(path, probeType)
        print("Calculating raw cryst 2 for " + probeType)
        Manager.calculateRawCryst2(path, probeType)
        print("Calculating raw cryst 3 for " + probeType)
        Manager.calculateRawCryst3(path, probeType)
        print("Calculating raw cryst 4 for " + probeType)
        Manager.calculateRawCryst4(path, probeType)
        print("Finished calculating crysts for " + probeType)

