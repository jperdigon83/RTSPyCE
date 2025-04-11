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

import bhmie
import numpy as np
import math as mt
from scipy.interpolate import interp1d

def integrate(a, func):
    
    funcC = np.copy(func)

    aRatio = a[1:] / a[:-1]

    idx = func < np.finfo(float).tiny
    func[idx] = np.finfo(float).tiny
    
    beta = np.log(funcC[..., 1:]/funcC[..., :-1]) / np.log(aRatio) + 1.

    dInt = funcC[..., :-1] * a[:-1] * (aRatio**beta - 1.) / beta

    return np.sum(dInt, axis=-1)


def opacities(wavelengths, refractive_index_list, grain_radii_list, grain_dist_list, grain_prop_list):
    """ Doc to write, but this function is better than the one below """

    assert np.sum(grain_prop_list) == 1.
    assert len(refractive_index_list) == len(grain_radii_list)
    assert len(refractive_index_list) == len(grain_dist_list)
    
    nnu = len(wavelengths)
    n_species = len(refractive_index_list)

    Cabs_species = np.empty((n_species, nnu))
    Csca_species = np.empty_like(Cabs_species)

    min_wave, max_wave = np.min(wavelengths*1e6), np.max(wavelengths*1e6)
    
    for i in range(n_species):

        wavemic, ncoef, kcoef = refractive_index_list[i][:, 0], refractive_index_list[i][:, 1], refractive_index_list[i][:, 2]

        # ---
        # Extrapolation of the refractive indices, if needed
        # ---
        
        if min_wave < np.min(wavemic):

            wavemic = np.append([min_wave], wavemic)
            ncoef = np.append([ncoef[0]], ncoef)
            kcoef = np.append([kcoef[0]], kcoef)
            

        if max_wave > np.max(wavemic):

            logwmaxwavemic = mt.log(max_wave/wavemic[-1])
            logwavemic = mt.log(wavemic[-1]/wavemic[-2])

            wavemic = np.append(wavemic, [max_wave])
            ncoef = np.append(ncoef, [ncoef[-1]*mt.exp(logwmaxwavemic*mt.log(ncoef[-1]/ncoef[-2])/logwavemic)])
            kcoef = np.append(kcoef, [kcoef[-1]*mt.exp(logwmaxwavemic*mt.log(kcoef[-1]/kcoef[-2])/logwavemic)])
            
        # ---
        # Interpolation of the refractive indices at the wavelenght grid
        # ---

        log_wave_mic = np.log(wavemic*1e-6)
        
        f = interp1d(log_wave_mic, np.log(ncoef))
        ncoefi = np.exp(f(np.log(wavelengths)))
    
        f = interp1d(log_wave_mic, np.log(kcoef))
        kcoefi = np.exp(f(np.log(wavelengths)))

        refidx = ncoefi + kcoefi * 1j

        # ---
        # Optical cross section for each dust species
        # ---
        
        if isinstance(grain_radii_list[i], (list, np.ndarray)):

            assert isinstance(grain_dist_list[i], (list, np.ndarray))
            
            # ---
            # Optical efficiencies for each dust species
            # ---
            
            na = len(grain_radii_list[i])
            x = 2 * mt.pi * grain_radii_list[i][None, :] / wavelengths[:, None]
            refidx = refidx[:, None] * np.ones((nnu, na))
            
            results = np.array([bhmie.bhmie(u, v, 2)[2:4] for u, v in zip(np.ravel(x), np.ravel(refidx))])
    
            Qext = results[:, 0].reshape((nnu, na))
            Qsca = results[:, 1].reshape((nnu, na))
    
            #
            # Averaging over the grains size distribution
            #
 
            K = mt.pi * grain_radii_list[i]**2 * grain_dist_list[i]
    
            Cext = integrate(grain_radii_list[i], K[None, :]*Qext) / integrate(grain_radii_list[i], grain_dist_list[i][None, :])

            Csca_species[i] = integrate(grain_radii_list[i], K[None, :]*Qsca) / integrate(grain_radii_list[i], grain_dist_list[i][None, :])
            Cabs_species[i] = Cext - Csca_species[i]
            
        else:

            x = 2 * mt.pi * grain_radii_list[i] / wavelengths

            results = np.array([bhmie.bhmie(x[k], refidx[k], 2)[2:4] for k in range(nnu)])

            Qext = results[:, 0]
            Qsca = results[:, 1]
    
            K = mt.pi * grain_radii_list[i]**2

            Cext = K * Qext
            
            Csca_species[i] = K * Qsca
            Cabs_species[i] = Cext - Csca_species[i]

    # ---
    # Averaging over all dust species
    # ---

    Cabs = np.sum(Cabs_species*grain_prop_list[:, None], axis=0)
    Csca = np.sum(Csca_species*grain_prop_list[:, None], axis=0)

    return Cabs, Csca
   
            
def dust_opacities(refidx_file, wavelengths, aGrains, wgt=None, extrapolate=False):
    """ My routine for computing opacities. Warning, this function is a lightweight version of the original one from the radmc3d package and does not contain any warnings or is not error safe. To use with many precautions."""

    wavmic, ncoef, kcoef = refidx_file[:, 0], refidx_file[:, 1], refidx_file[:, 2]
    
    #
    # Check range, and if needed and requested, extrapolate the
    # optical constants to longer or shorter wavelengths
    #
    
    if extrapolate:
        
        wmin, wmax = np.min(wavelengths)*1e6 * 0.999, np.max(wavelengths)*1e6 * 1.001
       
        if wmin < min(wavmic):

            ncoef = np.append([ncoef[0]], ncoef)
            kcoef = np.append([kcoef[0]], kcoef)
            wavmic = np.append([wmin], wavmic)
             
        if wmax > max(wavmic):
            
            logwmaxwavmic = mt.log(wmax/wavmic[-1])
            logwavemic = mt.log(wavmic[-1]/wavmic[-2])
                
            ncoef = np.append(ncoef, [ncoef[-1]*mt.exp(logwmaxwavmic*mt.log(ncoef[-1]/ncoef[-2])/logwavemic)])
                
            kcoef = np.append(kcoef, [kcoef[-1]*mt.exp(logwmaxwavmic*mt.log(kcoef[-1]/kcoef[-2])/logwavemic)])
            wavmic = np.append(wavmic, [wmax])
                       
    else:
        
        assert np.min(wavelengths) >= np.min(wavmic*1e-6), "Error: wavelength range out of range of the optical constants file.\n"
        
        assert np.max(wavelengths) <= np.max(wavmic*1e-6), "Error: wavelength range out of range of the optical constants file.\n"
   
    #
    # Interpolate
    # Note: Must be within range, otherwise stop
    #
    
    f = interp1d(np.log(wavmic*1e-6), np.log(ncoef))
    ncoefi = np.exp(f(np.log(wavelengths)))
    
    f = interp1d(np.log(wavmic*1e-6), np.log(kcoef))
    kcoefi = np.exp(f(np.log(wavelengths)))
    
    #
    # Make the complex index of refraction
    #
    
    refidx = ncoefi + kcoefi * 1j

    #
    # Compute x array
    #

    nl = len(wavelengths)

    if isinstance(aGrains, (list, tuple, np.ndarray)):
       
        if wgt is None:
            
            print("wgt set to None but aGrains is an array")
            
            raise ValueError
            
        na = len(aGrains)

        x = 2 * mt.pi * aGrains[None, :] / wavelengths[:, None]
    
        refidx = refidx[:, None] * np.ones((nl, na))

        results = np.array([bhmie.bhmie(u, v, 2)[2:4] for u, v in zip(np.ravel(x), np.ravel(refidx))])
    
        Qext = results[:, 0].reshape((nl, na))
        Qsca = results[:, 1].reshape((nl, na))
    
        #
        # Averaging over the grains size distribution
        #
 
        siggeom = mt.pi * aGrains * aGrains
    
        siggeomWgt = siggeom * wgt
    
        Cext = integrate(aGrains, Qext*siggeomWgt[None, :]) / integrate(aGrains, wgt[None, :])
    
        Csca = integrate(aGrains, Qsca*siggeomWgt[None, :]) / integrate(aGrains, wgt[None, :])

        # gsca = average(aGrains, gsca*wgt[None, :]) / average(aGrains, wgt[None, :])

    else:
        
        x = 2 * mt.pi * aGrains / wavelengths

        results = np.array([bhmie.bhmie(x[i], refidx[i])[2:4] for i in range(nl)])

        Qext = results[:, 0]
        Qsca = results[:, 1]
    
        siggeom = mt.pi * aGrains * aGrains

        Cext = siggeom * Qext

        Csca = siggeom * Qsca
   
    Cabs = Cext - Csca
    
    return Cabs, Csca

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    wave = np.logspace(-8, -3, 128)
    aGrains = np.logspace(-8, -5, 128)
    wgt = (aGrains)**(-3.5)

    refractive_indices_list = [np.genfromtxt("../data/dust/crMgFeSil.nk"), np.genfromtxt("../data/dust/amC_Hanner.nk")]

    grain_radii_list = [aGrains, aGrains]
    grain_dist_list = [wgt, wgt]
    grain_prop_list = np.array([1., 0.])

    Cabs, Csca = opacities(wave, refractive_indices_list, grain_radii_list, grain_dist_list, grain_prop_list)
    
    # Cabs, Csca = dust_opacities(refidx_file, wave, aGrains, wgt, extrapolate=True)

    plt.plot(wave, Cabs, label="amC-hanner-Cabs")
    plt.plot(wave, Csca, label="amC-hanner-Csca")
    
    # refidx_file = np.genfromtxt("../data/dust/amC-zb1.nk")
    # Cabs, Csca = dust_opacities(refidx_file, wave, aGrains, wgt, extrapolate=True)
    
    # plt.plot(wave, Cabs+Csca, label="amC-zb1")

    # refidx_file = np.genfromtxt("../data/dust/amC-zb2.nk")
    # Cabs, Csca = dust_opacities(refidx_file, wave, aGrains, wgt, extrapolate=True)
    
    # plt.plot(wave, Cabs+Csca, label="amC-zb2")

    # refidx_file = np.genfromtxt("../data/dust/amC-zb3.nk")
    # Cabs, Csca = dust_opacities(refidx_file, wave, aGrains, wgt, extrapolate=True)
    
    # plt.plot(wave, Cabs+Csca, label="amC-zb3")

    # refidx_file = np.genfromtxt("../data/dust/crMgFeSil.nk")
    # Cabs, Csca = dust_opacities(refidx_file, wave, aGrains, wgt, extrapolate=True)
    
    # plt.plot(wave, Cabs+Csca, label="crMgFeSil")

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
