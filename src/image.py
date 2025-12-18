import numpy as np
from envelope import Envelope
from source import BlackBodySphere, Source
from rtspyce import RTSpyce

class Image:

    def __init__(self, incl, PA, d, wave, x, y, dpix):

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

        if not PA >= 0.:
            raise ValueError("PA should be >= 0")

        if not PA < 360.:
            raise ValueError("PA should be < 360 degrees")

        self.incl = incl
        self.PA = PA
        self.d = d
        self.x = x
        self.y = y
        self.dpix = dpix
        self.wave = wave

        self.npix = len(self.x)
        self.nwave = len(self.wave)

        
    def compute_intensity(self, env: Envelope, src: Source):

        if not src.R == env.r[0]:
            raise ValueError("Source radius and r[0] should be equal")

        if not np.all(self.wave == env.wave):
            raise ValueError("Image and Envelope wavelengths should be equal")  # For the moment we impose this.
    
        if not np.all(self.wave == src.wave):
            raise ValueError("Image and Source wavelengths should be equal")
        
        foo = RTSpyce(env.r, env.theta)
        
        self.intensity = foo.intensity_map(self.x, self.y, self.incl, env.S, env.Kext, src.intensity)

        
    def compute_flux(self):

        return np.sum(self.intensity * self.dpix[None, :], axis=1) / self.d**2
    
    def compute_fourier_transform(self, u, v):

        alpha = self.x / self.d
        beta = self.y / self.d

        ulam = u[:, None] / self.wave[None, :]
        vlam = v[:, None] / self.wave[None, :]
        
        phasor = np.exp(-2.0 * 1j * mt.pi * (ulam[:, :, None] * alpha[None, None, :] + vlam[:, :, None] * beta[None, None, :]))
        
        return np.sum(self.intensity[None, :, :] * phasor * self.dpix[None, None, :], axis=-1) / self.d**2




class UniformCartesianImage(Image):

    def __init__(self, N, L, incl, PA, d, wave):

        if not L > 0:
            raise ValueError("L sould be positive")
        
        if not N > 0:
            raise ValueError("N should be positive")
        
        if not N % 2 == 0:
            raise ValueError("N should be even")
        
        self.N = N
        self.L = L
        
        xmax = 0.5 * L * (1. - 1./N)
        x = np.linspace(-xmax, xmax, N)

        x, y = np.meshgrid(x, x, indexing="ij")
        x = x.flatten()
        y = y.flatten()

        dpix = np.ones(len(x)) * L / N
    
        super().__init__(incl, PA, d, wave, x, y, dpix)

    def compute_intensity(self, env: Envelope, src: Source):

        if not src.R == env.r[0]:
            raise ValueError("Source radius and r[0] should be equal")

        if not np.all(self.wave == env.wave):
            raise ValueError("Image and Envelope wavelengths should be equal")
    
        if not np.all(self.wave == src.wave):
            raise ValueError("Image and Source wavelengths should be equal")
        
        foo = RTSpyce(env.r, env.theta)

        self.intensity = np.empty((self.nwave, self.npix))

        Nh = int(self.N**2/2)

        dum_var = foo.intensity_map(self.x[:Nh], self.y[:Nh], self.incl, env.S, env.Kext, src.intensity)

        self.intensity[:, :Nh] = dum_var
         
        dum_var = np.reshape(dum_var, (self.nwave, int(self.N/2), self.N))

        self.intensity[:, Nh:] = np.reshape(dum_var[:, ::-1, :], (self.nwave, (self.N*int(self.N/2))))

        
    def reconstruct_image(self):
        
        x = np.reshape(self.x, (self.N, self.N))
        y = np.reshape(self.y, (self.N, self.N))
        
        image = np.reshape(self.intensity, (self.nwave, self.N, self.N))
        
        return x, y, image

  
if __name__ == "__main__":

    import math as mt
    import matplotlib.pyplot as plt

    L = 20.
    nwave = 3
    N = 128
    incl = 60.
    PA = 0.
    d = 1.e9
    wave = np.linspace(1e-6, 10e-6, nwave)
    img = UniformCartesianImage(N, L, incl, PA, d, wave)

    
    nr, ntheta = 64, 64
    tau = 10.
    r = np.linspace(1., 10., nr)
    theta = np.linspace(0., 0.5*mt.pi, ntheta)
    Kext = tau * np.ones((nwave, nr, ntheta)) / (r[-1] - r[0])
    Kext[:, :, :-5] = 0.
    S = np.copy(Kext)
    env = Envelope(wave, r, theta, Kext, S)

    
    R = 1.0
    temp = 0.
    src = BlackBodySphere(R, temp, wave)

    img.compute_intensity(env, src)
    flux = img.compute_flux()
    
    u = np.linspace(15, 250, 128)
    ft = img.compute_fourier_transform(u, u)
    V2 = (np.abs(ft) / flux)**2
    print(np.shape(ft))
    
    x, y, intensity = img.reconstruct_image()
     
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 3.5), layout="tight")
   
    ax[0].pcolormesh(x, y, intensity[0])

    for i in range(nwave):
        ax[1].plot(2*u**2/wave[i], V2[:, i])

    ax[1].set_yscale("log")
    plt.show()
