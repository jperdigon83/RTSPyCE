import numpy as np
from envelope import Envelope
from source import Source
from rtspyce import RTSpyce

class Image:

    def __init__(self, incl, d, wave, x, y, dpix):

        if not len(x) == len(y):
            raise ValueError("x and y should have the same length")
        
        if not len(dpix) == len(x):
            raise ValueError("dpix x and y should have the same length")

        if not incl >= 0.:
            raise ValueError("incl should be positive of null")

        if not incl <= 0.:
            raise ValueError("incl should be inferior or equal to 90 degrees")
        
        if not d > 0.:
            raise ValueError("d should be positive")

        self.x = x
        self.y = y
        self.dpix = dpix
        self.incl = incl
        self.d = d
        self.wave = wave

        self.npix = len(self.x)
        self.nwave = len(self.wave)

    def compute_intensity(self, env: Envelope, src: Source):
        
        foo = RTSpyce(env.r, env.theta)
        
        self.intensity = foo.intensity_map(self.x, self.y, self.incl, env.S, env.Kext, src.intensity)
        
    def compute_flux(self):

        self.flux = np.sum(self.intensity * self.dpix[None, :], axis=1) / self.d**2
        

class UniformCartesianImage(Image):

    def __init__(self, N, L, incl, d, wave):

        self.N = N
        self.L = L
        dx = L / N
        xmax = 0.5 * L * (1. - dx)
        x = np.linspace(-xmax, xmax, N)
        x = np.repeat(x, len(x))

        dpix = np.ones(len(x)) * dx
    
        # super().__init__()

    def compute_intensity(self, env: Envelope, src: Source):
        
        foo = RTSpyce(env.r, env.theta)

        self.intensity = np.empty((self.npix, self.nwave))
        
        self.intensity[:, ] = foo.intensity_map(self.x[], self.x[], self.incl, env.S, env.Kext, src.intensity)
        
    def reconstruct_image(self):
        
        x = np.reshape(self.x, (self.N, self.N))
        image = np.reshape(self.intensity, (len(self.wavelengths), self.N, self.N))
        
        return x, image 

        


        

        

        


        
