import planck
import constants as ct
import os
from scipy.interpolate import interp1d
import math as mt
import numpy as np

class Source:

    def __init__(self, R, wave, intensity):

        if not R > 0.:
            raise ValueError("R should be positive")
        
        self.R = R
        self.wave = wave
        self.intensity = intensity

class BlackBody(Source):

    def __init__(self, R, Teff, wave):
        
        self.Teff = Teff
        nu = ct.c / wave
        intensity = planck.planck_function_freq(nu, Teff)
        
        super().__init__(R, wave, intensity)

class AtlasATM(Source):

    def __init__(self, name, R, Teff, logg, mh, wave):

        import warnings

        # Suppress all warnings
        warnings.filterwarnings("ignore")
        
        try:
            import pysynphot
        
        except ImportError:
            
            print("Error: The module could not be imported")

        self.Teff = Teff
        self.logg = logg
        self.mh = mh
        
        sp = pysynphot.Icat(name, Teff, mh, logg)
        sp.convert("m")
        sp.convert("fnu")

        # Convert flux to intensity in SI.
        
        intensity = 1e-3 * sp.flux / mt.pi
        
        intensity[intensity < np.finfo(float).tiny] = np.finfo(float).tiny

        interp = interp1d(np.log(sp.wave), np.log(intensity), fill_value="extrapolate")

        intensity = np.exp(interp(np.log(wave)))

        #interp = interp1d(np.log(sp.wave), intensity, fill_value="extrapolate")

        #intensity = interp(np.log(wave))

        super().__init__(R, wave, intensity)
