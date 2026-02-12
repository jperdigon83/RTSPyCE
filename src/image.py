import numpy as np
import math as mt
from envelope import Envelope
from source import Source
from rtspyce import RTSPyCE

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
        
        foo = RTSPyCE(env.r, env.theta)
        
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
        
        if not (N % 2) == 0:
            raise ValueError("N should be even")
        
        self.N = N
        self.L = L

        xmax = 0.5 * L
        xw = np.linspace(-xmax, xmax, N+1)
        x = 0.5 * (xw[1:] + xw[:-1])

        self.xw, self.yw = np.meshgrid(xw, xw, indexing="ij")

        x, y = np.meshgrid(x, x, indexing="ij")
        
        x = x.flatten()
        y = y.flatten()
        dpix = np.ones(len(x)) * (L / N)**2
    
        super().__init__(incl, PA, d, wave, x, y, dpix)

    def compute_intensity(self, env: Envelope, src: Source):

        if not src.R == env.r[0]:
            raise ValueError("Source radius and r[0] should be equal")

        if not np.all(self.wave == env.wave):
            raise ValueError("Image and Envelope wavelengths should be equal")
    
        if not np.all(self.wave == src.wave):
            raise ValueError("Image and Source wavelengths should be equal")

        if (2*src.R) <= (self.xw[1, 0] - self.xw[0, 0]):

            print("Warning: The star is not resolved in your image. You might want to add it after, in one of the central pixel, with the add_star() method.\n")

        self.rt = RTSPyCE(env.r, env.theta)

        self.intensity = np.empty((self.nwave, self.npix))

        Nh = int(self.N**2/2)

        dum_var = self.rt.intensity_map(self.x[:Nh], self.y[:Nh], self.incl, env.S, env.Kext, src.intensity)

        self.intensity[:, :Nh] = dum_var
         
        dum_var = np.reshape(dum_var, (self.nwave, int(self.N/2), self.N))

        self.intensity[:, Nh:] = np.reshape(dum_var[:, ::-1, :], (self.nwave, (self.N*int(self.N/2))))

        
    def add_star(self, env: Envelope, src: Source):

        print("Adding the star in one of the central pixel ...\n")
        
        origin = np.array([0.])
        
        star_intensity = self.rt.intensity_map(origin, origin, self.incl, env.S, env.Kext, src.intensity)
        
        idx = int(self.N/2 * (self.N+1))
       
        self.intensity[:, idx] += star_intensity.ravel() * mt.pi * src.R**2 / self.dpix[idx]

        
    def reconstruct_image(self):

        """Returns a copy of the structured cube of images """
        
        return np.reshape(self.intensity, (self.nwave, self.N, self.N))

    
class PolarImage(Image):
    
    def __init__(self, Rstar, nu_lin, R, nu_log, nv, incl, PA, d, wave):

        if not (nv % 2) == 0:
            raise ValueError("nv should be even")

        if not Rstar > 0.:
            raise ValueError("Rstar should be strictly positive")

        if not R > Rstar:
            raise ValueError("R should be larger than Rstar")

        self.nu = nu_lin + nu_log
        self.nv = nv
        
        vw = np.linspace(0., 2*mt.pi, nv+1)
        v = 0.5 * (vw[1:] + vw[:-1])
        dv = vw[1:] - vw[:-1]

        uw_lin = np.linspace(0., Rstar, nu_lin+1)
        u_lin = 0.5 * (uw_lin[1:] + uw_lin[:-1])
        
        aw = np.linspace(0., mt.log(R/Rstar), nu_log+1)
        a = 0.5 * (aw[1:] + aw[:-1])
        uw_log = Rstar * np.exp(aw)
        u_log = Rstar * np.exp(a)

        uw = np.concatenate((uw_lin[:-1], uw_log))
        u = np.concatenate((u_lin, u_log))
      
        self.vw, self.uw = np.meshgrid(vw, uw, indexing="ij")
        
        x = u[None, :] * np.sin(v[:, None])
        y = u[None, :] * np.cos(v[:, None])

        dpix = 0.5 * dv[:, None] * (uw[None, 1:]**2 - uw[None, :-1]**2)
        
        x = x.flatten()
        y = y.flatten()
        dpix = dpix.flatten()

        super().__init__(incl, PA, d, wave, x, y, dpix)

    def compute_intensity(self, env: Envelope, src: Source):

        if not src.R == env.r[0]:
            raise ValueError("Source radius and r[0] should be equal")

        if not np.all(self.wave == env.wave):
            raise ValueError("Image and Envelope wavelengths should be equal")
    
        if not np.all(self.wave == src.wave):
            raise ValueError("Image and Source wavelengths should be equal")
        
        foo = RTSPyCE(env.r, env.theta)

        self.intensity = np.empty((self.nwave, self.npix))

        Nh = int(self.nu*self.nv/2)

        dum_var = foo.intensity_map(self.x[:Nh], self.y[:Nh], self.incl, env.S, env.Kext, src.intensity)

        self.intensity[:, :Nh] = dum_var
         
        dum_var = np.reshape(dum_var, (self.nwave, int(self.nv/2), self.nu))

        self.intensity[:, Nh:] = np.reshape(dum_var[:, ::-1, :], (self.nwave, (self.nu*int(self.nv/2))))


    def reconstruct_image(self):
        
        return np.reshape(self.intensity, (self.nwave, self.nv, self.nu))
