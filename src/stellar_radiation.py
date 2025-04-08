# * ----       This file is part of the project        ----
# * 
# * Copyright (C) 2024 Perdigon, J.. All Rights Reserved.
# * 
# * This file is licensed under the terms of the GNU       
# * General Public License, version 3., as published by the
# * Free Software Foundation. 
# * 
# * This file is distributed in the hope that it will be   
# * useful, but WITHOUT ANY WARRANTY; without even the     
# * implied warranty of MERCHANTABILITY or FITNESS FOR A   
# * PARTICULAR PURPOSE. See the GNU General Public License 
# * for more details.
# * 
# * You should have received a copy of the GNU General     
# * Public License along with this program. If not, see    
# * <https://www.gnu.org/licenses/>.

import os
import planck
from scipy.interpolate import interp1d
import numpy as np
import math as mt
import constants as ct

try:
        
    import pysynphot as S
    os.environ['PYSYN_CDBS']
            
except ImportError:
            
    print("Error: The module could not be imported.")

    

def stellar_radiation(wavelengths, name="blackbody", teff=5772 , mh=0.0122, logg=4.438):

    name_list = np.array(["blackbody", "ck04models", "k93models", "phoenix"])
    
    assert name in name_list

    if name != "blackbody":
        
        try:
            
            sp = S.Icat(name, teff, mh, logg)
            sp.convert("m")
            sp.convert("flam")

            I_star = sp.flux * 1e7 / mt.pi
            I_star[I_star < np.finfo(float).tiny] = np.finfo(float).tiny

            interp = interp1d(np.log(sp.wave), np.log(I_star), fill_value="extrapolate")

            I_star = np.exp(interp(np.log(wavelengths)))
        
            I_star[I_star != I_star] = 0.

        except S.exceptions.ParameterOutOfBounds:
            
            print("Stellar parameters out of the grid, falling back to blackbody spectrum")
            
            I_star = planck.planck_function_wave(wavelengths, teff)

        return I_star

    else:

        return planck.planck_function_wave(wavelengths, teff)
        
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    wave = np.logspace(-7.5, -6, 256)
    Teff = 40000.

    logg = np.linspace(2.0, 5.0, 10)

    for logg in logg:
        
        I_star = stellar_radiation(wave, "ck04models", teff=Teff, logg=logg)
        
        plt.plot(wave, I_star)

    bb = stellar_radiation(wave, "blackbody", teff=Teff)
    
    plt.plot(wave, bb)
  
    
    plt.xscale("log")
    plt.yscale("log")

    plt.show()

    
    

        

        
