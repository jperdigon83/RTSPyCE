import planck
import constants as ct

class Source:

    def __init__(self, R, nu, intensity):

        if not R > 0.:
            raise ValueError("R should be positive")
        
        self.R = R
        self.nu = nu
        self.wave = ct.c / nu
        self.intensity = intensity

class BlackBodySphere(Source):

    def __init__(self, R, temp, nu):

        self.temp = temp
        
        intensity = planck.planck_function_freq(nu, temp)
        
        super().__init__(R, nu, intensity)

        
