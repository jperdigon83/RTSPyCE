import numpy as np
import math as mt

class Envelope:

    def __init__(self, wave, r, theta, Kext, S):

        if not np.all(wave > 0.):
            raise ValueError("wavelengths should be positive")

        if not np.all(np.diff(wave) > 0.):
            raise ValueError("wavelengths should be strictly increasing")
        
        if not np.all(r > 0.):
            raise ValueError("r should be positive")

        if not np.all(np.diff(r) > 0.):
            raise ValueError("r should be strictly increasing")
        
        if not theta[0] == 0.:
            raise ValueError("theta[0] should be 0")

        if not theta[-1] == 0.5*mt.pi:
            raise ValueError("theta[-1] should be 0.5*mt.pi")

        if not np.all(np.diff(theta) > 0.):
            raise ValueError("theta should be strictly increasing")

        nwave = len(wave)
        nr = len(r)
        ntheta = len(theta)
        
        if not np.shape(Kext) == (nwave, nr, ntheta):
            raise ValueError("Kext should be of shape (nnu, nr, ntheta)")
        
        if not np.all(Kext >= 0.):
            raise ValueError("Kext should be positive or zero")

        if not np.shape(S) == (nwave, nr, ntheta):
            raise ValueError("S should be of shape (nnu, nr, ntheta)")
        
        if not np.all(S >= 0.):
            raise ValueError("S should be positive or zero")
    
        self.wave = wave
        self.r = r
        self.theta = theta
        self.Kext = Kext
        self.S = S
