import matplotlib.pyplot as plt
import matplotlib as mlt
import numpy as np
from src.spectra import Spectra
import os 
from src.gallPeaks import GallPeaks

class Graph:
    dpi = 250

    def __init__(self, dpi):
        self.dpi = dpi 

    def plotGraph(self, limits, legend, spectra: Spectra, path, override):
        if limits == ():
            limits = (200, 3400)
        if legend == []:
            legend = ["widmo"]
        graphPath = path + spectra.name + ".png"
        if os.path.exists(graphPath) and (not override):
            return
        mlt.use("Cairo")
        figure, axis = plt.subplots()
        axis.set_ylabel("Intensywność [j.u.]")
        axis.set(yticklabels = []) # removing tick labels
        axis.set_xlabel("Przesunięcie Ramana [cm$^{-1}$]")
        plt.xlim(limits)
        plt.plot(spectra.shifts, spectra.intensities)
        plt.legend(legend)
        plt.gca().invert_xaxis()
        plt.savefig(path + spectra.name + ".png", dpi=self.dpi)
        plt.close()


    def plotGraphWithGallPeaks(self, path: str, spectraName: str):
        peaks = GallPeaks()
        figure, axis = plt.subplots()
        plt.plot(self.xData, self.yData)
        plt.xlim((250, 3400))
        for peakPosition in peaks.peaks:
            plt.plot([peakPosition, peakPosition], [np.min(self.yData), np.max(self.yData)], linewidth=0.5)
        plt.gca().invert_xaxis()
        plt.savefig(path + "gall_" + spectraName + ".png", dpi = self.dpi)
        plt.close()
