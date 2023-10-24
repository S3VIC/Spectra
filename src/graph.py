import matplotlib.pyplot as plt
import matplotlib as mlt
import numpy as np
from src.spectra import Spectra
import os 
from src.gallPeaks import GallPeaks

class Graph:
    dpi = 250
    name = ""
    x = []
    y = []

    def __init__(self, dpi: int, spectra: Spectra):
        self.dpi = dpi
        x = spectra.shifts
        y = spectra.intensities
        name = spectra.name

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
        plt.close()
