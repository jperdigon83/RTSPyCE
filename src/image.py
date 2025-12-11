import numpy as np
from envelope import Envelope
from source import BlackBodySphere, Source
from rtspyce import RTSpyce

class Image:

    def __init__(self, incl, d, wave, x, y, dpix):

        if not len(x) == len(y):
            raise ValueError("x and y should have the same length")
        
        if not len(dpix) == len(x):
            raise ValueError("dpix x and y should have the same length")

        if not np.all(dpix > 0.):
            raise ValueError("Elements of dpix should be positive")
            
        if not incl >= 0.:
            raise ValueError("incl should be positive of null")

        if not incl <= 90.:
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

        if not L > 0:
            raise ValueError("L sould be positive")
        
        if not N > 0:
            raise ValueError("N should be positive")
        
        if not N % 2 == 0:
            raise ValueError("N should be even")
        
        self.N = N
        self.L = L
        
        dx = L / N
        xmax = 0.5 * L * (1. - dx)
        x = np.linspace(-xmax, xmax, N)

        x, y = np.meshgrid(x, x, indexing="ij")
        x = x.flatten()
        y = y.flatten()

        dpix = np.ones(len(x)) * dx
    
        super().__init__(incl, d, wave, x, y, dpix)

    def compute_intensity(self, env: Envelope, src: Source):
        
        foo = RTSpyce(env.r, env.theta)

        self.intensity = np.empty((self.nwave, self.npix))

        Nh = int(self.N**2/2)
        
        self.intensity[:, :Nh] = foo.intensity_map(self.x[:Nh], self.y[:Nh], self.incl, env.S, env.Kext, src.intensity)

        self.intensity[:, Nh:] = self.intensity[:, :Nh][:, ::-1]
    
        
    def reconstruct_image(self):
        
        x = np.reshape(self.x, (self.N, self.N))
        y = np.reshape(self.y, (self.N, self.N))
        
        image = np.reshape(self.intensity, (self.nwave, self.N, self.N))
        
        return x, y, image 



    
if __name__ == "__main__":

    import math as mt
    import matplotlib.pyplot as plt

    L = 5.
    nwave = 2
    N = 32
    wave = np.linspace(1e-6, 2e-6, nwave)
    img = UniformCartesianImage(N, L, 0., 10., wave)

    
    nr, ntheta = 16, 16

    r = np.linspace(1., 10., nr)
    theta = np.linspace(0., 0.5*mt.pi, ntheta)
    Kext = np.ones((nwave, nr, ntheta))
    S = np.copy(Kext)
    env = Envelope(wave, r, theta, Kext, S)

    R = 1.0
    temp = 5800.
    src = BlackBodySphere(R, temp, wave)

    img.compute_intensity(env, src)

    x, y, intensity = img.reconstruct_image()
    
    print(x, y, intensity)

    plt.pcolormesh(x, y, intensity[0])
    plt.show()
