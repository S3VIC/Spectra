

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

