import numpy as np

class Image:

    def __init__(self, x, y, dpix, incl, d, wave, intensity):

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
        self.intensity = intensity
        self.flux = np.sum(intensity * dpix[None, :], axis=1) / d**2
        

class UniformCartesianImage(Image):

    def __init__(self, N, L, incl, d, wavelengths):

        self.N = N
        self.L = L
        dx = L / N
        xmax = 0.5 * L * (1. - dx)
        x = np.linspace(-xmax, xmax, N)
        x = np.repeat(x, len(x))

        dpix = np.ones(len(x)) * dx
    
        #super().__init__(x, y, dpix, incl, d, wavelengths, intensity)

    def reconstruct_image(self):
        
        x = np.reshape(self.x, (self.N, self.N))
        image = np.reshape(self.intensity, (len(self.wavelengths), self.N, self.N))
        
        return x, image 

        


        

        

        


        
