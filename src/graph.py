import matplotlib.pyplot as plt
import matplotlib as mlt
import numpy as np
from src.spectra import Spectra
import os 
from src.gallPeaks import GallPeaks
from src.logger import Logger


class Graph:
    dpi = 250
    name = ""
    x = []
    y = []

    def __init__(self, dpi: int, spectra: Spectra):
        self.dpi = dpi
        self.x = spectra.shifts
        self.y = spectra.intensities
        self.name = spectra.name

    def plotGraph(self, limits, legend, path, override):
        if limits == ():
            limits = (200, 3400)
        if not legend:
            legend = ["widmo"]
        graphPath = path + self.name + ".png"
        if os.path.exists(graphPath) and (not override):
            return
        mlt.use("Cairo")
        figure, axis = plt.subplots()
        axis.set_ylabel("Intensywność [j.u.]")
        axis.set(yticklabels = [])  # removing tick labels
        axis.set_xlabel("Przesunięcie Ramana [cm$^{-1}$]")
        plt.xlim(limits)
        plt.plot(self.x, self.y)
        plt.legend(legend)
        plt.gca().invert_xaxis()
        plt.savefig(path + self.name + ".png", dpi=self.dpi)
        Logger.logInfo(message = "Saved graph " + graphPath)
        plt.close()

    def plotGraphWithGallPeaks(self, path: str):
        peaks = GallPeaks()
        figure, axis = plt.subplots()
        plt.plot(self.x, self.y)
        plt.xlim((250, 3400))
        for peakPosition in peaks.peaks:
            plt.plot([peakPosition, peakPosition], [np.min(self.y), np.max(self.x)], linewidth=0.5)
        plt.gca().invert_xaxis()
        plt.savefig(path + "gall_" + self.name + ".png", dpi = self.dpi)
        Logger.logInfo(message = "Saved graph for " + self.name)
        plt.close()

    def plotGraphAsLS(self, spectra: Spectra, path: str):
        legend = ["widmo", "widmo poprawione", "linia bazowa"]
        mlt.use("Cairo")
        figure, axis = plt.subplots()
        axis.set_ylabel("Intensywność [j.u.]")
        axis.set(yticklabels = [])  # removing tick labels
        axis.set_xlabel("Przesunięcie Ramana [cm$^{-1}$]")
        plt.plot(spectra.shifts, spectra.intensities)
        plt.plot(spectra.shifts, spectra.asLSIntensities)
        plt.plot(spectra.shifts, spectra.intensities - spectra.asLSIntensities)
        plt.legend(legend)
        plt.gca().invert_xaxis()
        graphPath = path + "asLS-" + self.name + ".png"
        plt.savefig(graphPath, dpi = self.dpi)
        print("Saved graph to: " + graphPath)
        plt.close()

    def plotGraphArLS(self, spectra: Spectra, path: str):
        legend = ["widmo", "widmo poprawione", "linia bazowa"]
        mlt.use("Cairo")
        figure, axis = plt.subplots()
        axis.set_ylabel("Intensywność [j.u.]")
        axis.set(yticklabels = [])  # removing tick labels
        axis.set_xlabel("Przesunięcie Ramana [cm$^{-1}$]")
        plt.plot(spectra.shifts, spectra.intensities)
        plt.plot(spectra.shifts, spectra.asLSIntensities)
        plt.plot(spectra.shifts, spectra.intensities - spectra.asLSIntensities)
        plt.legend(legend)
        plt.gca().invert_xaxis()
        graphPath = path + "arLS-" + self.name + ".png"
        plt.savefig(graphPath, dpi = self.dpi)
        print("Saved graph to: " + graphPath)
        plt.close()

    def plotGraphAsLSArLSCombined(self, spectra: Spectra, path: str):
        legend = ("widmo", "asLS", "arLS")
        mlt.use("Cairo")
        figure, axis = plt.subplots()
        axis.set_ylabel("Intensywność [j.u.]")
        axis.set(yticklabels = [])  # removing tick labels
        axis.set_xlabel("Przesunięcie Ramana [cm$^{-1}$]")
        plt.plot(spectra.shifts, spectra.intensities)
        asLSbaseline = spectra.intensities - spectra.asLSIntensities
        arLSbaseline = spectra.intensities - spectra.arLSIntensities
        plt.plot(spectra.shifts, asLSbaseline)
        plt.plot(spectra.shifts, arLSbaseline)
        plt.legend(legend)
        plt.gca().invert_xaxis()
        graphPath = path + "comparison-" + self.name + ".png"
        plt.savefig(graphPath, dpi = self.dpi)
        print("Saved comparison graph to: " + graphPath)
        plt.close()

    def plotDeconvFit(self, spectra: Spectra, fit, path: str, override: bool):
        if not os.path.exists(path):
            os.mkdir(path)
        if os.path.exists(path + self.name + ".png") and not override:
            return
        legend = ("widmo", "dopasowanie")
        mlt.use("Cairo")
        figure, axis = plt.subplots()
        axis.set_ylabel("Intensywność [j.u.]")
        axis.set(yticklabels=[])  # removing tick labels
        axis.set_xlabel("Przesunięcie Ramana [cm$^{-1}$]")
        plt.plot(spectra.shifts, spectra.intensities)
        plt.plot(spectra.shifts, fit)
        plt.legend(legend)
        plt.gca().invert_xaxis()
        graphPath = path + self.name + ".png"
        plt.savefig(graphPath, dpi=self.dpi)
        print("Saved deconv fit graph to: " + graphPath)
        plt.close()
