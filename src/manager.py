from src.spectra import Spectra
from src.fileManager import FileManager
from src.graph import Graph
import numpy as np
from typing import List
import scipy as sc
from scipy.optimize import curve_fit
import os
from src.parameters import GaussParams, LorentzParams
import src.modelFunctions as fn
from src.parameters import DiagnosticSignals
import src.analyzer as an

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
    def saveAsLSCorrection(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool, lamb: str, term: str):
        dirPath = os.path.join(path, dirName)
        dirPath = dirPath + lamb + "/" + term + "/"
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
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
    def saveArLSCorrection(spectra: Spectra, path: str, dirName: str, override: bool, plot: bool, lamb: str, weight: str):
        dirPath = os.path.join(path, dirName)
        dirPath = dirPath + lamb + "/" + weight + "/"
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
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
    def deconv(path: str, lambdaRange, asLSSecond, arLSSecond, rootOutputPath):
        Manager.deconvStretch(path=path, correctionMethod="asLS", lambdaRange = lambdaRange, secondParamRange = asLSSecond, rootOutputPath = rootOutputPath)
        Manager.deconvStretch(path=path, correctionMethod="arLS", lambdaRange = lambdaRange, secondParamRange = arLSSecond, rootOutputPath = rootOutputPath)
        #Manager.deconvBend(path=path, correctionMethod="asLS", lambdaRange, asLSSecond)
        #Manager.deconvBend(path=path, correctionMethod="arLS", lambdaRange, arLSSecond)
        #Manager.deconvTwist(path=path, correctionMethod="asLS", lambdaRange, asLSSecond)
        #Manager.deconvTwist(path=path, correctionMethod="arLS", lambdaRange, arLSSecond)
        #Manager.deconvCcstretch(path=path, correctionMethod="asLS", lambdaRange, asLSSecond)
        #Manager.deconvCcstretch(path=path, correctionMethod="arLS", lambdaRange, arLSSecond)

    @staticmethod
    def deconvStretch(path: str, correctionMethod: str, lambdaRange, secondParamRange, rootOutputPath):
        vibrationDir = "stretch"
        gaussName = "gauss"
        lorentzName = "lorentz"
        for lambdaParam in lambdaRange:
            for secondParam in secondParamRange:
                spectrasPath = os.path.join(path, vibrationDir, correctionMethod, str(lambdaParam), str(secondParam) + "/")
                gaussDeconvPath = os.path.join(rootOutputPath, gaussName)
                lorentzDeconvPath = os.path.join(spectrasPath, lorentzName)
                if not os.path.exists(gaussDeconvPath):
                    os.makedirs(gaussDeconvPath)
                if not os.path.exists(lorentzDeconvPath):
                    os.makedirs(lorentzDeconvPath)
                print("Started deconvolution for " + spectrasPath)
                fileManager = FileManager(path=spectrasPath)
                spectraList = Manager.getSpectraList(manager=fileManager)
                gaussParams = GaussParams()
                lorentzParams = LorentzParams()
                funcGauss = fn.gauss4term
                funcLorentz = fn.lorentz4term
                for spectra in spectraList:
                    #if os.path.exists(spectrasPath + "gauss/" + spectra.name + ".CSV") and os.path.exists(
                    #        spectrasPath + "gauss/" + spectra.name + ".CSV"):
                    #    continue
                    try:
                        poptGauss = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                              p0=gaussParams.c1_inits, bounds=gaussParams.c1_bounds)[0]
                        fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3],
                                             poptGauss[4],
                                             poptGauss[5], poptGauss[6], poptGauss[7], poptGauss[8], poptGauss[9],
                                             poptGauss[10],
                                             poptGauss[11])

                        poptLorentz = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                                p0=lorentzParams.c1_inits, bounds=lorentzParams.c1_bounds)[0]
                        fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                                 poptLorentz[4], poptLorentz[5], poptLorentz[6], poptLorentz[7], poptLorentz[8],
                                                 poptLorentz[9], poptLorentz[10], poptLorentz[11])

                        gauss1 = fn.Gauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2])
                        gauss2 = fn.Gauss(spectra.shifts, poptGauss[3], poptGauss[4], poptGauss[5])
                        gauss3 = fn.Gauss(spectra.shifts, poptGauss[6], poptGauss[7], poptGauss[8])
                        gauss4 = fn.Gauss(spectra.shifts, poptGauss[9], poptGauss[10], poptGauss[11])
                        gaussFilePath = os.path.join(gaussDeconvPath, str(lambdaParam), str(secondParam) + "/")
                        if not os.path.exists(gaussFilePath):
                            os.makedirs(gaussFilePath)
                        gaussFile = open(gaussFilePath + spectra.name + ".CSV", "w")
                        for index in range(len(spectra.shifts)):
                            gaussFile.write(str(spectra.shifts[index]) + "," + str(gauss1[index]) + "," + str(gauss2[index]) + "," +
                                            str(gauss3[index]) + "," + str(gauss4[index]) +
                                            "," + str(poptGauss[0]) + "," + str(poptGauss[1]) + "," + str(poptGauss[2]) + ","
                                            + str(poptGauss[3]) + "," + str(poptGauss[4]) + "," + str(poptGauss[5]) +
                                            "," + str(poptGauss[6]) + "," + str(poptGauss[7]) + "," + str(poptGauss[8]) + ","
                                            + str(poptGauss[9]) + "," + str(poptGauss[10]) + "," + str(poptGauss[11]) + '\n')

                        lorentz1 = fn.Lorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2])
                        lorentz2 = fn.Lorentz(spectra.shifts, poptLorentz[3], poptLorentz[4], poptLorentz[5])
                        lorentz3 = fn.Lorentz(spectra.shifts, poptLorentz[6], poptLorentz[7], poptLorentz[8])
                        lorentz4 = fn.Lorentz(spectra.shifts, poptLorentz[9], poptLorentz[10], poptLorentz[11])
                        lorentzFilePath = os.path.join(gaussDeconvPath, str(lambdaParam), str(secondParam) + "/")
                        if not os.path.exists(lorentzFilePath):
                            os.makedirs(lorentzFilePath)
                        lorentzFile = open(lorentzFilePath + spectra.name + ".CSV", "w")
                        for index in range(len(spectra.shifts)):
                            lorentzFile.write(
                                str(spectra.shifts[index]) + "," + str(lorentz1[index]) + "," + str(lorentz2[index]) + "," +
                                str(lorentz3[index]) + "," + str(lorentz4[index]) +
                                "," + str(poptLorentz[0]) + "," + str(poptLorentz[1]) + "," + str(poptLorentz[2]) + ","
                                + str(poptLorentz[3]) + "," + str(poptLorentz[4]) + "," + str(poptLorentz[5]) +
                                "," + str(poptLorentz[6]) + "," + str(poptLorentz[7]) + "," + str(poptLorentz[8]) + ","
                                + str(poptLorentz[9]) + "," + str(poptLorentz[10]) + "," + str(poptLorentz[11]) + '\n')

                    except RuntimeError:
                        print("Optimal parameters not found for " + spectra.name + ", continuing")
                        continue

                    graph = Graph(spectra=spectra, dpi=250)
                    graph.plotDeconvFit(spectra=spectra, fit=fitGauss, path=gaussFilePath, override=False)
                    graph.plotDeconvFit(spectra=spectra, fit=fitLorentz, path=lorentzFilePath, override=False)

                print("Finished deconvolution for " + spectrasPath)

    @staticmethod
    def deconvBend(path: str, correctionMethod: str):
        vibrationDir = "bend/"
        spectrasPath = os.path.join(path, vibrationDir + correctionMethod + "/")
        gaussDeconvPath = spectrasPath + "gauss/"
        lorentzDeconvPath = spectrasPath + "lorentz/"
        if not os.path.exists(gaussDeconvPath):
            os.mkdir(gaussDeconvPath)
        if not os.path.exists(lorentzDeconvPath):
            os.mkdir(lorentzDeconvPath)
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss4term
        funcLorentz = fn.lorentz4term
        print("Performing bend deconvolution")
        for spectra in spectraList:
            #if os.path.exists(spectrasPath + "gauss/" + spectra.name + ".CSV") and os.path.exists(
            #        spectrasPath + "gauss/" + spectra.name + ".CSV"):
            #    continue
            try:
                poptGauss = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                      p0=gaussParams.c2_inits, bounds=gaussParams.c2_bounds)[0]
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3],
                                     poptGauss[4],
                                     poptGauss[5], poptGauss[6], poptGauss[7], poptGauss[8], poptGauss[9],
                                     poptGauss[10],
                                     poptGauss[11])

                poptLorentz = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                        p0=lorentzParams.c2_inits, bounds=lorentzParams.c2_bounds)[0]
                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5], poptLorentz[6], poptLorentz[7], poptLorentz[8],
                                         poptLorentz[9], poptLorentz[10], poptLorentz[11])
                gauss1 = fn.Gauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2])
                gauss2 = fn.Gauss(spectra.shifts, poptGauss[3], poptGauss[4], poptGauss[5])
                gauss3 = fn.Gauss(spectra.shifts, poptGauss[6], poptGauss[7], poptGauss[8])
                gauss4 = fn.Gauss(spectra.shifts, poptGauss[9], poptGauss[10], poptGauss[11])
                gaussFile = open(spectrasPath + "gauss/" + spectra.name + ".CSV", "w")
                for index in range(len(spectra.shifts)):
                    gaussFile.write(str(spectra.shifts[index]) + "," + str(gauss1[index]) + "," + str(gauss2[index]) + "," +
                                    str(gauss3[index]) + "," + str(gauss4[index]) +
                                    "," + str(poptGauss[0]) + "," + str(poptGauss[1]) + "," + str(poptGauss[2]) + ","
                                    + str(poptGauss[3]) + "," + str(poptGauss[4]) + "," + str(poptGauss[5]) +
                                    "," + str(poptGauss[6]) + "," + str(poptGauss[7]) + "," + str(poptGauss[8]) + ","
                                    + str(poptGauss[9]) + "," + str(poptGauss[10]) + "," + str(poptGauss[11]) + '\n')

                lorentz1 = fn.Lorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2])
                lorentz2 = fn.Lorentz(spectra.shifts, poptLorentz[3], poptLorentz[4], poptLorentz[5])
                lorentz3 = fn.Lorentz(spectra.shifts, poptLorentz[6], poptLorentz[7], poptLorentz[8])
                lorentz4 = fn.Lorentz(spectra.shifts, poptLorentz[9], poptLorentz[10], poptLorentz[11])
                lorentzFile = open(spectrasPath + "lorentz/" + spectra.name + ".CSV", "w")
                for index in range(len(spectra.shifts)):
                    lorentzFile.write(
                        str(spectra.shifts[index]) + "," + str(lorentz1[index]) + "," + str(lorentz2[index]) + "," +
                        str(lorentz3[index]) + "," + str(lorentz4[index]) +
                        "," + str(poptLorentz[0]) + "," + str(poptLorentz[1]) + "," + str(poptLorentz[2]) + ","
                        + str(poptLorentz[3]) + "," + str(poptLorentz[4]) + "," + str(poptLorentz[5]) +
                        "," + str(poptLorentz[6]) + "," + str(poptLorentz[7]) + "," + str(poptLorentz[8]) + ","
                        + str(poptLorentz[9]) + "," + str(poptLorentz[10]) + "," + str(poptLorentz[11]) + '\n')
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
        gaussDeconvPath = spectrasPath + "gauss/"
        lorentzDeconvPath = spectrasPath + "lorentz/"
        if not os.path.exists(gaussDeconvPath):
            os.mkdir(gaussDeconvPath)
        if not os.path.exists(lorentzDeconvPath):
            os.mkdir(lorentzDeconvPath)
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss2term
        funcLorentz = fn.lorentz2term
        print("Performing ccstretch deconvolution")
        for spectra in spectraList:
            #if os.path.exists(spectrasPath + "gauss/" + spectra.name + ".CSV") and os.path.exists(
            #        spectrasPath + "gauss/" + spectra.name + ".CSV"):
            #    continue
            try:
                poptGauss = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                      p0=gaussParams.c3_inits, bounds=gaussParams.c3_bounds)[0]
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3],
                                     poptGauss[4],
                                     poptGauss[5])

                poptLorentz = curve_fit(f=funcLorentz, xdata=spectra.shifts, ydata=spectra.intensities,
                                        p0=lorentzParams.c3_inits, bounds=lorentzParams.c3_bounds)[0]
                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5])

                gauss1 = fn.Gauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2])
                gauss2 = fn.Gauss(spectra.shifts, poptGauss[3], poptGauss[4], poptGauss[5])
                gaussFile = open(spectrasPath + "gauss/" + spectra.name + ".CSV", "w")
                for index in range(len(spectra.shifts)):
                    gaussFile.write(str(spectra.shifts[index]) + "," + str(gauss1[index]) + "," + str(gauss2[index]) +
                                    "," + str(poptGauss[0]) + "," + str(poptGauss[1]) + "," + str(poptGauss[2]) + ","
                                    + str(poptGauss[3]) + "," + str(poptGauss[4]) + "," + str(poptGauss[5]) + '\n')

                lorentz1 = fn.Lorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2])
                lorentz2 = fn.Lorentz(spectra.shifts, poptLorentz[3], poptLorentz[4], poptLorentz[5])
                lorentzFile = open(spectrasPath + "lorentz/" + spectra.name + ".CSV", "w")
                for index in range(len(spectra.shifts)):
                    lorentzFile.write(
                        str(spectra.shifts[index]) + "," + str(lorentz1[index]) + "," + str(lorentz2[index]) +
                        "," + str(poptLorentz[0]) + "," + str(poptLorentz[1]) + "," + str(poptLorentz[2]) + ","
                        + str(poptLorentz[3]) + "," + str(poptLorentz[4]) + "," + str(poptLorentz[5]) + '\n')
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
        gaussDeconvPath = spectrasPath + "gauss/"
        lorentzDeconvPath = spectrasPath + "lorentz/"
        if not os.path.exists(gaussDeconvPath):
            os.mkdir(gaussDeconvPath)
        if not os.path.exists(lorentzDeconvPath):
            os.mkdir(lorentzDeconvPath)
        fileManager = FileManager(path=spectrasPath)
        spectraList = Manager.getSpectraList(manager=fileManager)
        gaussParams = GaussParams()
        lorentzParams = LorentzParams()
        funcGauss = fn.gauss2term
        funcLorentz = fn.lorentz2term
        print("Performing ccstretch deconvolution")
        for spectra in spectraList:
            #if os.path.exists(spectrasPath + "gauss/" + spectra.name + ".CSV") and os.path.exists(
            #        spectrasPath + "gauss/" + spectra.name + ".CSV"):
            #    continue
            try:
                poptGauss = curve_fit(f=funcGauss, xdata=spectra.shifts, ydata=spectra.intensities,
                                      p0=gaussParams.c4_inits, bounds=gaussParams.c4_bounds)[0]
                fitGauss = funcGauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2], poptGauss[3],
                                     poptGauss[4],
                                     poptGauss[5])

                poptLorentz = curve_fit(f=funcLorentz, xdata=spectra.shifts,
                                        ydata=spectra.intensities,
                                        p0=lorentzParams.c4_inits, bounds=lorentzParams.c4_bounds)[0]

                fitLorentz = funcLorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2], poptLorentz[3],
                                         poptLorentz[4], poptLorentz[5])

                gauss1 = fn.Gauss(spectra.shifts, poptGauss[0], poptGauss[1], poptGauss[2])
                gauss2 = fn.Gauss(spectra.shifts, poptGauss[3], poptGauss[4], poptGauss[5])
                gaussFile = open(spectrasPath + "gauss/" + spectra.name + ".CSV", "w")
                for index in range(len(spectra.shifts)):
                    gaussFile.write(str(spectra.shifts[index]) + "," + str(gauss1[index]) + "," + str(gauss2[index]) +
                                    "," + str(poptGauss[0]) + "," + str(poptGauss[1]) + "," + str(poptGauss[2]) + ","
                                    + str(poptGauss[3]) + "," + str(poptGauss[4]) + "," + str(poptGauss[5]) + '\n')

                lorentz1 = fn.Lorentz(spectra.shifts, poptLorentz[0], poptLorentz[1], poptLorentz[2])
                lorentz2 = fn.Lorentz(spectra.shifts, poptLorentz[3], poptLorentz[4], poptLorentz[5])
                lorentzFile = open(spectrasPath + "lorentz/" + spectra.name + ".CSV", "w")
                for index in range(len(spectra.shifts)):
                    lorentzFile.write(
                        str(spectra.shifts[index]) + "," + str(lorentz1[index]) + "," + str(lorentz2[index]) +
                        "," + str(poptLorentz[0]) + "," + str(poptLorentz[1]) + "," + str(poptLorentz[2]) + ","
                        + str(poptLorentz[3]) + "," + str(poptLorentz[4]) + "," + str(poptLorentz[5]) + '\n')

            except RuntimeError:
                print("Optimal parameters not found, continuing")
                continue

            graph = Graph(spectra=spectra, dpi=250)
            graph.plotDeconvFit(spectra=spectra, fit=fitGauss, path=spectrasPath + "gauss/", override=False)
            graph.plotDeconvFit(spectra=spectra, fit=fitLorentz, path=spectrasPath + "lorentz/", override=False)

    @staticmethod
    def calculateRawCryst1(path: str, probeType: str, param1: str, param2: str, pathToSave: str, method: str):
        fileName = "cryst1-" + probeType + "-" + param1 + "-" + param2 + "-raw.CSV"
        if os.path.exists(fileName):
            return
        spectrasDir = "stretch/" + method + param1 + "/" + param2 + "/"
        spectrasPath = os.path.join(path, spectrasDir)
        fileManager = FileManager(spectrasPath)
        spectraList = Manager.getSpectraList(fileManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_str_sym'], SignalsDict['CH3_str_asym']])
        if not os.path.exists(pathToSave):
            os.makedirs(pathToSave)
        filePath = os.path.join(pathToSave, fileName)
        file = open(filePath, "w")
        for spectra in spectraList:
            intensitiesForCryst = np.array([])
            spectra.findPeaks()
            for signal in signals:
                peakPosition = spectra.findPeakDifferences(signal=signal)[1]
                intensitiesForCryst = np.append(intensitiesForCryst,
                                                spectra.intensities[np.where(spectra.shifts == peakPosition)])
            cryst = intensitiesForCryst[0] / intensitiesForCryst[1]
            file.write(spectra.name + '$' + str(cryst) + '\n')
        file.close()

    @staticmethod
    def calculateRawCryst2(path: str, probeType: str, param1: str, param2: str, pathToSave: str, method: str):
        fileName = "cryst2-" + probeType + "-" + param1 + "-" + param2 + "-raw.CSV"
        if os.path.exists(fileName):
            return
        spectrasDir = "bend/" + method + param1 + "/" + param2 + "/"
        spectrasPath = os.path.join(path, spectrasDir)
        fileManager = FileManager(spectrasPath)
        spectraList = Manager.getSpectraList(fileManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_ben_cryst'], SignalsDict['CH2_ben_amorf']])
        filePath = os.path.join(pathToSave, fileName)
        if not os.path.exists(pathToSave):
            os.makedirs(pathToSave)
        file = open(filePath, "w")
        for spectra in spectraList:
            intensitiesForCryst = np.array([])
            spectra.findPeaks()
            for signal in signals:
                peakPosition = spectra.findPeakDifferences(signal=signal)[1]
                intensitiesForCryst = np.append(intensitiesForCryst,
                                                spectra.intensities[np.where(spectra.shifts == peakPosition)])
            cryst = intensitiesForCryst[0] / intensitiesForCryst[1]
            file.write(spectra.name + '$' + str(cryst) + '\n')
        file.close()

    @staticmethod
    def calculateRawCryst3(path: str, probeType: str, param1: str, param2: str, pathToSave: str, method: str):
        fileName = "cryst3-" + probeType + "-" + param1 + "-" + param2 + "-raw.CSV"
        if os.path.exists(fileName):
            return
        spectrasDir1 = "bend/" + method + param1 + "/" + param2 + "/"
        spectrasDir2 = "twist/" + method + param1 + "/" + param2 + "/"
        spectrasPath1 = os.path.join(path, spectrasDir1)
        spectrasPath2 = os.path.join(path, spectrasDir2)
        bendSpectrasManager = FileManager(spectrasPath1)
        twistSpectrasManager = FileManager(spectrasPath2)
        bendSpectraList = Manager.getSpectraList(bendSpectrasManager)
        twistSpectraList = Manager.getSpectraList(twistSpectrasManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_ben_cryst'], SignalsDict['CH2_twist_amorf']])
        filePath = os.path.join(pathToSave, fileName)
        if not os.path.exists(pathToSave):
            os.makedirs(pathToSave)
        file = open(filePath, "w")
        for spectraTwist in twistSpectraList:
            for spectraBend in bendSpectraList:
                if spectraBend.name == spectraTwist.name:
                    spectraBend.findPeaks()
                    spectraTwist.findPeaks()
                    bendPeakPosition = spectraBend.findPeakDifferences(signal=signals[0])[1]
                    twistPeakPosition = spectraTwist.findPeakDifferences(signal=signals[1])[1]
                    bendIntensity = spectraBend.intensities[np.where(spectraBend.shifts == bendPeakPosition)]
                    twistIntensity = spectraTwist.intensities[np.where(spectraTwist.shifts == twistPeakPosition)]
                    cryst = bendIntensity / twistIntensity
                    if str(cryst) == '':
                        break
                    file.write(spectraBend.name + '$' + str(cryst).replace("[", "").replace("]", "") + '\n')
        file.close()

    @staticmethod
    def calculateRawCryst4(path: str, probeType: str, param1: str, param2: str, pathToSave: str, method: str):
        fileName = "cryst4-" + probeType + "-" + param1 + "-" + param2 + "-raw.CSV"
        if os.path.exists(fileName):
            return
        spectrasDir1 = "bend/" + method + param1 + "/" + param2 + "/"
        spectrasDir2 = "ccstretch/" + method + param1 + "/" + param2 + "/"
        spectrasPath1 = os.path.join(path, spectrasDir1)
        spectrasPath2 = os.path.join(path, spectrasDir2)
        bendSpectrasManager = FileManager(spectrasPath1)
        ccstretchSpectrasManager = FileManager(spectrasPath2)
        bendSpectraList = Manager.getSpectraList(bendSpectrasManager)
        ccstretchSpectraList = Manager.getSpectraList(ccstretchSpectrasManager)
        SignalsDict = DiagnosticSignals.getSignalsDict()
        signals = np.array([SignalsDict['CH2_ben_cryst'], SignalsDict['CC_str_amorf']])
        filePath = os.path.join(pathToSave, fileName)
        if not os.path.exists(pathToSave):
            os.makedirs(pathToSave)
        file = open(filePath, "w")
        for spectraCcstretch in ccstretchSpectraList:
            for spectraBend in bendSpectraList:
                if spectraBend.name == spectraCcstretch.name:
                    spectraBend.findPeaks()
                    spectraCcstretch.findPeaks()
                    bendPeakPosition = spectraBend.findPeakDifferences(signal=signals[0])[1]
                    ccStretchPeakPosition = spectraCcstretch.findPeakDifferences(signal=signals[1])[1]
                    bendIntensity = spectraBend.intensities[np.where(spectraBend.shifts == bendPeakPosition)]
                    ccStretchIntensity = spectraCcstretch.intensities[
                        np.where(spectraCcstretch.shifts == ccStretchPeakPosition)]
                    cryst = bendIntensity / ccStretchIntensity
                    print(type(cryst))
                    file.write(spectraBend.name + '$' + str(cryst).replace("[", "").replace("]", "") + '\n')
        file.close()

    @staticmethod
    def calculateRawCrysts(path: str, probeType: str, param1: str, param2: str, pathToSave: str, method: str):
        Manager.calculateRawCryst1(path, probeType, param1, param2, pathToSave, method)
        Manager.calculateRawCryst2(path, probeType, param1, param2, pathToSave, method)
        Manager.calculateRawCryst3(path, probeType, param1, param2, pathToSave, method)
        Manager.calculateRawCryst4(path, probeType, param1, param2, pathToSave, method)


    @staticmethod
    def calculateDeconvCryst1(path: str, probeType: str):
        rootCrystsFolder = "crysts/"
        numberFolder = "1/"
        correctionMethodsFolders = ["asLS/", "arLS/"]
        deconvMethodsFolders = ["gauss/", "lorentz/"]
        rootFolder = os.path.join(path, "stretch/")
        for correctionMethod in correctionMethodsFolders:
            correctionPath = os.path.join(rootFolder, correctionMethod)
            for deconvMethod in deconvMethodsFolders:
                finalPath = rootCrystsFolder + probeType + correctionMethod + deconvMethod + numberFolder
                if not os.path.exists(finalPath):
                    os.makedirs(finalPath)
                fileRectLeft = open(finalPath + "rectLeft.CSV", "w")
                fileRectRight = open(finalPath + "rectRight.CSV", "w")
                fileTrap = open(finalPath + "trap.CSV", "w")
                deconvMethodPath = os.path.join(correctionPath, deconvMethod)
                spectrasManager = FileManager(deconvMethodPath)
                for file in spectrasManager.fileList:
                    data = np.loadtxt(spectrasManager.path + file, delimiter=",")
                    shifts = data[:, 0]
                    model1 = data[:, 1]
                    model2 = data[:, 2]
                    area1RectLeft = an.rectIntegLeft(shifts, model1)
                    area2RectLeft = an.rectIntegLeft(shifts, model2)
                    crystRectLeft = area1RectLeft / area2RectLeft

                    area1RectRight = an.rectIntegRight(shifts, model1)
                    area2RectRight = an.rectIntegRight(shifts, model2)
                    crystRectRight = area1RectRight / area2RectRight

                    area1trap = an.trapInteg(shifts, model1)
                    area2trap = an.trapInteg(shifts, model2)
                    crystTrap = area1trap / area2trap

                    fileRectLeft.write(file[:-4] + ";" + str(crystRectLeft) + '\n')
                    fileRectRight.write(file[:-4] + ";" + str(crystRectRight) + '\n')
                    fileTrap.write(file[:-4] + ";" + str(crystTrap) + '\n')
                fileRectLeft.close()
                fileRectRight.close()
                fileTrap.close()


    @staticmethod
    def calculateDeconvCryst2(path: str, probeType: str):
        rootCrystsFolder = "crysts/"
        numberFolder = "2/"
        correctionMethodsFolders = ["asLS/", "arLS/"]
        deconvMethodsFolders = ["gauss/", "lorentz/"]
        rootFolder = os.path.join(path, "bend/")
        for correctionMethod in correctionMethodsFolders:
            correctionPath = os.path.join(rootFolder, correctionMethod)
            for deconvMethod in deconvMethodsFolders:
                finalPath = rootCrystsFolder + probeType + correctionMethod + deconvMethod + numberFolder
                if not os.path.exists(finalPath):
                    os.mkdir(finalPath)
                fileRectLeft = open(finalPath + "rectLeft.CSV", "w")
                fileRectRight = open(finalPath + "rectRight.CSV", "w")
                fileTrap = open(finalPath + "trap.CSV", "w")
                deconvMethodPath = os.path.join(correctionPath, deconvMethod)
                spectrasManager = FileManager(deconvMethodPath)
                for file in spectrasManager.fileList:
                    data = np.loadtxt(spectrasManager.path + file, delimiter=",")
                    shifts = data[:, 0]
                    model1 = data[:, 1]
                    model2 = data[:, 2]
                    area1RectLeft = an.rectIntegLeft(shifts, model1)
                    area2RectLeft = an.rectIntegLeft(shifts, model2)
                    crystRectLeft = area1RectLeft / area2RectLeft

                    area1RectRight = an.rectIntegRight(shifts, model1)
                    area2RectRight = an.rectIntegRight(shifts, model2)
                    crystRectRight = area1RectRight / area2RectRight

                    area1trap = an.trapInteg(shifts, model1)
                    area2trap = an.trapInteg(shifts, model2)
                    crystTrap = area1trap / area2trap

                    fileRectLeft.write(file[:-4] + ";" + str(crystRectLeft) + '\n')
                    fileRectRight.write(file[:-4] + ";" + str(crystRectRight) + '\n')
                    fileTrap.write(file[:-4] + ";" + str(crystTrap) + '\n')
                fileRectLeft.close()
                fileRectRight.close()
                fileTrap.close()

    @staticmethod
    def calculateDeconvCryst3(path: str, probeType: str):
        rootCrystsFolder = "crysts/"
        numberFolder = "3/"
        correctionMethodsFolders = ["asLS/", "arLS/"]
        deconvMethodsFolders = ["gauss/", "lorentz/"]
        twistRootFolder = os.path.join(path, "twist/")
        bendRootFolder = os.path.join(path, "bend/")
        for correctionMethod in correctionMethodsFolders:
            twistCorrectionPath = os.path.join(twistRootFolder, correctionMethod)
            bendCorrectionPath = os.path.join(bendRootFolder, correctionMethod)
            for deconvMethod in deconvMethodsFolders:
                finalPath = rootCrystsFolder + probeType + correctionMethod + deconvMethod + numberFolder
                if not os.path.exists(finalPath):
                    os.mkdir(finalPath)
                fileRectLeft = open(finalPath + "rectLeft.CSV", "w")
                fileRectRight = open(finalPath + "rectRight.CSV", "w")
                fileTrap = open(finalPath + "trap.CSV", "w")
                twistDeconvMethodPath = os.path.join(twistCorrectionPath, deconvMethod)
                bendDeconvMethodPath = os.path.join(bendCorrectionPath, deconvMethod)
                twistSpectrasManager = FileManager(twistDeconvMethodPath)
                bendSpectrasManager = FileManager(bendDeconvMethodPath)
                for twistFile in twistSpectrasManager.fileList:
                    for bendFile in bendSpectrasManager.fileList:
                        if twistFile != bendFile:
                            continue
                        twistData = np.loadtxt(twistSpectrasManager.path + twistFile, delimiter = ",")
                        bendData = np.loadtxt(bendSpectrasManager.path + bendFile, delimiter = ",")
                        twistShifts = twistData[:, 0]
                        bendShifts = bendData[:, 0]
                        model1 = bendData[:, 1]
                        model2 = twistData[:, 1]
                        area1RectLeft = an.rectIntegLeft(bendShifts, model1)
                        area2RectLeft = an.rectIntegLeft(twistShifts, model2)
                        crystRectLeft = area1RectLeft / area2RectLeft

                        area1RectRight = an.rectIntegRight(bendShifts, model1)
                        area2RectRight = an.rectIntegRight(twistShifts, model2)
                        crystRectRight = area1RectRight / area2RectRight

                        area1trap = an.trapInteg(bendShifts, model1)
                        area2trap = an.trapInteg(twistShifts, model2)
                        crystTrap = area1trap / area2trap

                        fileRectLeft.write(twistFile[:-4] + ";" + str(crystRectLeft) + '\n')
                        fileRectRight.write(twistFile[:-4] + ";" + str(crystRectRight) + '\n')
                        fileTrap.write(twistFile[:-4] + ";" + str(crystTrap) + '\n')
                        break
                fileRectLeft.close()
                fileRectRight.close()
                fileTrap.close()

    @staticmethod
    def calculateDeconvCryst4(path: str, probeType: str):
        rootCrystsFolder = "crysts/"
        numberFolder = "4/"
        correctionMethodsFolders = ["asLS/", "arLS/"]
        deconvMethodsFolders = ["gauss/", "lorentz/"]
        ccstretchRootFolder = os.path.join(path, "ccstretch/")
        bendRootFolder = os.path.join(path, "bend/")
        for correctionMethod in correctionMethodsFolders:
            ccstretchCorrectionPath = os.path.join(ccstretchRootFolder, correctionMethod)
            bendCorrectionPath = os.path.join(bendRootFolder, correctionMethod)
            for deconvMethod in deconvMethodsFolders:
                finalPath = rootCrystsFolder + probeType + correctionMethod + deconvMethod + numberFolder
                if not os.path.exists(finalPath):
                    os.mkdir(finalPath)
                fileRectLeft = open(finalPath + "rectLeft.CSV", "w")
                fileRectRight = open(finalPath + "rectRight.CSV", "w")
                fileTrap = open(finalPath + "trap.CSV", "w")
                ccstretchDeconvMethodPath = os.path.join(ccstretchCorrectionPath, deconvMethod)
                bendDeconvMethodPath = os.path.join(bendCorrectionPath, deconvMethod)
                ccstretchSpectrasManager = FileManager(ccstretchDeconvMethodPath)
                bendSpectrasManager = FileManager(bendDeconvMethodPath)
                for ccstretchFile in ccstretchSpectrasManager.fileList:
                    for bendFile in bendSpectrasManager.fileList:
                        if ccstretchFile != bendFile:
                            continue
                        ccstretchData = np.loadtxt(ccstretchSpectrasManager.path + ccstretchFile, delimiter = ",")
                        bendData = np.loadtxt(bendSpectrasManager.path + bendFile, delimiter = ",")
                        ccstretchShifts = ccstretchData[:, 0]
                        bendShifts = bendData[:, 0]
                        model1 = bendData[:, 1]
                        model2 = ccstretchData[:, 1]
                        area1RectLeft = an.rectIntegLeft(bendShifts, model1)
                        area2RectLeft = an.rectIntegLeft(ccstretchShifts, model2)
                        crystRectLeft = area1RectLeft / area2RectLeft

                        area1RectRight = an.rectIntegRight(bendShifts, model1)
                        area2RectRight = an.rectIntegRight(ccstretchShifts, model2)
                        crystRectRight = area1RectRight / area2RectRight

                        area1trap = an.trapInteg(bendShifts, model1)
                        area2trap = an.trapInteg(ccstretchShifts, model2)
                        crystTrap = area1trap / area2trap

                        fileRectLeft.write(ccstretchFile[:-4] + ";" + str(crystRectLeft) + '\n')
                        fileRectRight.write(ccstretchFile[:-4] + ";" + str(crystRectRight) + '\n')
                        fileTrap.write(ccstretchFile[:-4] + ";" + str(crystTrap) + '\n')
                        break
                fileRectLeft.close()
                fileRectRight.close()
                fileTrap.close()

    @staticmethod
    def calculateDeconvCrysts(path: str, probeType: str):
        Manager.calculateDeconvCryst1(path = path, probeType = probeType)
        Manager.calculateDeconvCryst2(path=path, probeType=probeType)
        Manager.calculateDeconvCryst3(path=path, probeType=probeType)
        Manager.calculateDeconvCryst4(path=path, probeType=probeType)
