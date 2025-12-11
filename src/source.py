import planck

class Source:

    def __init__(self, R, wave, intensity):

        if not R > 0.:
            raise ValueError("R should be positive")
        
        self.R = R
        self.wave = wave
        self.intensity = intensity

class BlackBodySphere(Source):

    def __init__(self, R, temp, wave):

        self.temp = temp
        
        intensity = planck.planck_function_wave(wave, temp)
        
        super().__init__(R, wave, intensity)
