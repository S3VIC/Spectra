import numpy as np

class DiagnosticSignals:
    def __init__(self):
        pass

    @staticmethod
    def getSignalsDict():
        return {
            'CH3_str_asym':  2882,
            'CH2_str_sym': 2848,
            'CH2_ben_amorf':  1440,
            'CH2_ben_cryst':  1416,
            'CH2_twist_amorf':  1303,
            'CC_str_amorf':  1080
        }

class SpectraFragments:
    rangeNames = []
    rangeLimits = []
    signals = {}

    def __init__(self):
        self.rangeNames = ["stretch", "bend", "twist", "ccstretch"]
        self.rangeLimits = [(2780, 2990), (1380, 1500), (1270, 1350), (1000, 1200)]
        self.signals = {
                'CH3_str_asym' : (1, 2882),
                'CH2_str_sym' : (1, 2848),
                'CH2_ben_amorf' : (2, 1440),
                'CH2_ben_cryst' : (2, 1416),
                'CH2_twist_amorf' : (3, 1303),
                'CC_str_amorf' : (4, 1080)
                }


class LorentzParams:
    def __init__(self):
        self.c1_bounds = None
        self.c1_inits = None
        self.c2_bounds = None
        self.c2_inits = None
        self.c3_bounds = None
        self.c3_inits = None
        self.c4_bounds = None
        self.c4_inits = None
        self.setParams()

    def setParams(self):
        self.setParamsForCryst1Lorentz()
        self.setParamsForCryst2Lorentz()
        self.setParamsForCryst3Lorentz()
        self.setParamsForCryst4Lorentz()

    def setParamsForCryst1Lorentz(self):
        spectraParams = SpectraFragments()
        p1_min = np.array([0.01, spectraParams.signals['CH2_str_sym'][1] - 20, 0.5], dtype = 'float')
        p2_min = np.array([0.01, spectraParams.signals['CH3_str_asym'][1] - 20, 0.5], dtype = 'float')
        p3_min = np.array([0.01, 2880, 0.5], dtype = 'float')
        p4_min = np.array([0.01, 2925, 0.5], dtype = 'float')

        p1_max = np.array([np.inf, spectraParams.signals['CH2_str_sym'][1] + 20, np.inf], dtype = 'float')
        p2_max = np.array([np.inf, spectraParams.signals['CH3_str_asym'][1] + 20, np.inf], dtype = 'float')
        p3_max = np.array([np.inf, 2912, np.inf], dtype = 'float')
        p4_max = np.array([np.inf, 2939, np.inf], dtype = 'float')

        c1_bounds_low = np.concatenate((p1_min, p2_min, p3_min, p4_min), axis=None)
        c1_bounds_high = np.concatenate((p1_max, p2_max, p3_max, p4_max), axis=None)

        p1_init = np.array([1, spectraParams.signals['CH2_str_sym'][1], 1], dtype='float')
        p2_init = np.array([1, spectraParams.signals['CH3_str_asym'][1], 1], dtype='float')
        p3_init = np.array([1, 2905, 1], dtype='float')
        p4_init = np.array([1, 2932, 1], dtype='float')

        self.c1_bounds = (c1_bounds_low, c1_bounds_high)
        self.c1_inits = np.concatenate((p1_init, p2_init, p3_init, p4_init), axis=None)

    def setParamsForCryst2Lorentz(self):
        spectraParams = SpectraFragments()
        p1_min = np.array([0.001, spectraParams.signals['CH2_ben_cryst'][1] - 15, 0.1], dtype='float')
        p2_min = np.array([0.001, spectraParams.signals['CH2_ben_amorf'][1] - 15, 0.1], dtype='float')
        p3_min = np.array([0.001, 1450, 1], dtype='float')
        p4_min = np.array([0.001, 1470, 1], dtype='float')

        p1_max = np.array([np.inf, spectraParams.signals['CH2_ben_cryst'][1] + 15, np.inf], dtype='float')
        p2_max = np.array([np.inf, spectraParams.signals['CH2_ben_amorf'][1] + 15, np.inf], dtype='float')
        p3_max = np.array([np.inf, 1474, np.inf], dtype='float')
        p4_max = np.array([np.inf, 1485, np.inf], dtype='float')

        c2_bounds_low = np.concatenate((p1_min, p2_min, p3_min, p4_min), axis=None)
        c2_bounds_high = np.concatenate((p1_max, p2_max, p3_max, p4_max), axis=None)
        # init values

        p1_init = np.array([1, spectraParams.signals['CH2_ben_cryst'][1], 1], dtype='float')
        p2_init = np.array([1, spectraParams.signals['CH2_ben_amorf'][1], 1], dtype='float')
        p3_init = np.array([1, 1460, 1], dtype='float')
        p4_init = np.array([1, 1475, 1], dtype='float')

        self.c2_bounds = (c2_bounds_low, c2_bounds_high)
        self.c2_inits = np.concatenate((p1_init, p2_init, p3_init, p4_init), axis=None)

    def setParamsForCryst3Lorentz(self):
        spectraParams = SpectraFragments()
        p1_min = np.array([0.01, spectraParams.signals['CH2_twist_amorf'][1] - 15, 1], dtype='float')
        p2_min = np.array([0.01, 1310, 1], dtype='float')

        p1_max = np.array([np.inf, spectraParams.signals['CH2_twist_amorf'][1] + 1, np.inf], dtype='float')
        p2_max = np.array([np.inf, 1319, np.inf], dtype='float')

        c3_bounds_low = np.concatenate((p1_min, p2_min), axis=None)
        c3_bounds_high = np.concatenate((p1_max, p2_max), axis=None)

        p1_init = np.array([1, spectraParams.signals['CH2_twist_amorf'][1], 3], dtype='float')
        p2_init = np.array([1, 1315, 3], dtype='float')

        self.c3_bounds = (c3_bounds_low, c3_bounds_high)
        self.c3_inits = np.concatenate((p1_init, p2_init), axis=None)

    def setParamsForCryst4Lorentz(self):
        # remember to add 3rd peak
        spectraParams = SpectraFragments()
        p1_min = np.array([1, spectraParams.signals['CC_str_amorf'][1] - 5, 1], dtype='float')
        p2_min = np.array([1, 1065, 1], dtype='float')

        p1_max = np.array([np.inf, spectraParams.signals['CC_str_amorf'][1] + 5, np.inf], dtype='float')
        p2_max = np.array([np.inf, 1075, np.inf], dtype='float')

        c4_bounds_low = np.concatenate((p1_min, p2_min), axis=None)
        c4_bounds_high = np.concatenate((p1_max, p2_max), axis=None)

        # init values
        p1_init = np.array([1, spectraParams.signals['CC_str_amorf'][1], 1], dtype='float')
        p2_init = np.array([1, 1070, 1], dtype='float')

        self.c4_bounds = (c4_bounds_low, c4_bounds_high)
        self.c4_inits = np.concatenate((p1_init, p2_init), axis=None)


class GaussParams:
    def __init__(self):
        self.c1_bounds = None
        self.c1_inits = None
        self.c2_bounds = None
        self.c2_inits = None
        self.c3_bounds = None
        self.c3_inits = None
        self.c4_bounds = None
        self.c4_inits = None
        self.setParamsForCryst1()
        self.setParamsForCryst2()
        self.setParamsForCryst3()
        self.setParamsForCryst4()

    def setParamsForCryst1(self):
        spectraParams = SpectraFragments()
        p1_min = np.array([0.5, spectraParams.signals['CH2_str_sym'][1] - 10, 1], dtype='float')
        p2_min = np.array([0.5, spectraParams.signals['CH3_str_asym'][1] - 10, 1], dtype='float')
        p3_min = np.array([0.5, 2883, 1], dtype='float')
        p4_min = np.array([0.5, 2923, 1], dtype='float')

        p1_max = np.array([4.5e6, spectraParams.signals['CH2_str_sym'][1] + 10, 20], dtype='float')
        p2_max = np.array([4.5e6, spectraParams.signals['CH3_str_asym'][1] + 10, 20], dtype='float')
        p3_max = np.array([1.5e5, 2915, 30], dtype='float')
        p4_max = np.array([1.5e5, 2945, 30], dtype='float')

        c1_bounds_low = np.concatenate((p1_min, p2_min, p3_min, p4_min), axis=None)
        c1_bounds_high = np.concatenate((p1_max, p2_max, p3_max, p4_max), axis=None)

        # init values
        p1_init = np.array([1, spectraParams.signals['CH2_str_sym'][1], 1], dtype='float')
        p2_init = np.array([1, spectraParams.signals['CH3_str_asym'][1], 1], dtype='float')
        p3_init = np.array([1, 2905, 1], dtype='float')
        p4_init = np.array([1, 2932, 1], dtype='float')

        self.c1_bounds = (c1_bounds_low, c1_bounds_high)
        self.c1_inits = np.concatenate((p1_init, p2_init, p3_init, p4_init), axis=None)

    def setParamsForCryst2(self):
        spectraParams = SpectraFragments()
        p1_min = np.array([1, spectraParams.signals['CH2_ben_cryst'][1] - 15, 1], dtype='float')
        p2_min = np.array([1, spectraParams.signals['CH2_ben_amorf'][1] - 15, 1], dtype='float')
        p3_min = np.array([0, 1449, 2], dtype='float')
        p4_min = np.array([1, 1460, 2], dtype='float')

        p1_max = np.array([4.5e5, spectraParams.signals['CH2_ben_cryst'][1] + 10, 20], dtype='float')
        p2_max = np.array([4.5e5, spectraParams.signals['CH2_ben_amorf'][1] + 10, 20], dtype='float')
        p3_max = np.array([4.5e5, 1468, 20], dtype='float')
        p4_max = np.array([4.5e5, 1485, 20], dtype='float')

        c2_bounds_low = np.concatenate((p1_min, p2_min, p3_min, p4_min), axis=None)
        c2_bounds_high = np.concatenate((p1_max, p2_max, p3_max, p4_max), axis=None)

        # init values
        p1_inits = np.array([1, spectraParams.signals['CH2_ben_cryst'][1], 3], dtype='float')
        p2_inits = np.array([1, spectraParams.signals['CH2_ben_amorf'][1], 3], dtype='float')
        p3_inits = np.array([1, 1460, 3], dtype='float')
        p4_inits = np.array([1, 1475, 3], dtype='float')

        self.c2_bounds = (c2_bounds_low, c2_bounds_high)
        self.c2_inits = np.concatenate((p1_inits, p2_inits, p3_inits, p4_inits), axis=None)

    def setParamsForCryst3(self):
        spectraParams = SpectraFragments()
        p1_min = np.array([1, spectraParams.signals['CH2_twist_amorf'][1] - 4, 1], dtype='float')
        p2_min = np.array([1, 1280, 1], dtype='float')  # medit

        p1_max = np.array([4.5e3, spectraParams.signals['CH2_twist_amorf'][1] + 4, 15], dtype='float')
        p2_max = np.array([4.5e3, 1319, 8], dtype='float')  # medit

        c3_bounds_low = np.concatenate((p1_min, p2_min), axis=None)
        c3_bounds_high = np.concatenate((p1_max, p2_max), axis=None)

        # init values
        p1_inits = np.array([1, spectraParams.signals['CH2_twist_amorf'][1], 1], dtype='float')
        p2_inits = np.array([1, 1300, 1], dtype='float')  # medit

        self.c3_bounds = (c3_bounds_low, c3_bounds_high)
        self.c3_inits = np.concatenate((p1_inits, p2_inits), axis=None)

    def setParamsForCryst4(self):
        spectraParams = SpectraFragments()
        c4_min_p1 = np.array([1, spectraParams.signals['CC_str_amorf'][1] - 8, 1], dtype='float')
        c4_min_p2 = np.array([1, 1045, 1], dtype='float')  # medit

        c4_max_p1 = np.array([4.5e3, spectraParams.signals['CC_str_amorf'][1] + 8, 15], dtype='float')
        c4_max_p2 = np.array([4.5e3, 1075, 4], dtype='float')  # medit

        c4_bounds_low = np.concatenate((c4_min_p1, c4_min_p2), axis=None)
        c4_bounds_high = np.concatenate((c4_max_p1, c4_max_p2), axis=None)

        # init values
        c4_init_p1 = np.array([1, spectraParams.signals['CC_str_amorf'][1], 1], dtype='float')
        c4_init_p2 = np.array([1, 1060, 1], dtype='float')

        self.c4_bounds = (c4_bounds_low, c4_bounds_high)
        self.c4_inits = np.concatenate((c4_init_p1, c4_init_p2), axis=None)
