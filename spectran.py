#!/usr/bin/python3
import os
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
                for name, values in fragments.signals.items():
                    if values[0] == index + 1:
                        dataFileName = name + "_rs_stability.CSV"
                        newFilePath = dataPath + folder + dataFileName

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
