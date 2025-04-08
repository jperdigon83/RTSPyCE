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

import numpy as np
import astropy.constants as ct

# ---
# CONSTANTS
# ---

c2 = ct.c.value * ct.c.value
hc = ct.h.value * ct.c.value
hc2 = ct.h.value * c2
dhc2 = 2 * hc2
dhc = 2 * hc

dhoc2 = 2. * ct.h.value / c2

hkB = ct.h.value / ct.k_B.value
hckB = hkB * ct.c.value

# ---
# FUNCTIONS
# ---

def planck_function_freq(freq, T):

    if hasattr(T, "__len__"):

        num_new_axes = len(np.shape(T))
    
        for _ in range(num_new_axes):

            freq = np.expand_dims(freq, axis=-1)

        freq = np.tile(freq, np.shape(T))

        T = np.expand_dims(T, axis=0) * np.ones((np.shape(freq)[0], *T.shape))
  
    with np.errstate(divide='ignore'):
        
        X = ct.h.value * freq / (ct.k_B.value * T)

    nu3 = freq * freq * freq
    
    idx = X < 1e-5

    B = np.empty_like(X)

    if not hasattr(T, "__len__"):

        T = np.tile(T, np.shape(freq)[0])
        
    B[idx] = dhc * (T[idx] / hckB)**3 * X[idx]*X[idx] * (1. - X[idx])

    expXm = np.exp(-X[~idx])
    
    B[~idx] = dhoc2 * nu3[~idx] * expXm / (1 - expXm)
    
    return B

def planck_function_wave(wave, T):

    if hasattr(T, "__len__"):

        num_new_axes = len(np.shape(T))
    
        for _ in range(num_new_axes):

            wave = np.expand_dims(wave, axis=-1)
            
        wave = np.tile(wave, np.shape(T))

        T = np.expand_dims(T, axis=0) * np.ones((np.shape(wave)[0], *T.shape))

    with np.errstate(divide='ignore'):
        X = hckB / (wave * T)
        
    idx = X < 1e-5
    
    B = np.empty_like(X)

    if not hasattr(T, "__len__"):
        
        T = np.tile(T, np.shape(wave)[0])

    B[idx] = dhc2 * (T[idx] / hckB)**5 * X[idx]**4 * (1 - X[idx])

    expX = np.exp(-X[~idx])
    
    B[~idx] = dhc2 / wave[~idx]**5 * expX / (1 - expX)

    return B
    
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.close('all')
    fontsize = 20

    lam = np.logspace(-15, 15., 1024)
    nu = ct.c.value / lam

    # ---
    # 1ST TEST: The whole temperature range
    # ---

    T = np.array([0., 1e-3, 1e1, 1e2, 1e3, 5e2, 1e4, 2e4, 5e4, 1e5, 1e6])
    
    Bnu = planck_function_freq(nu, T)
    Blam = planck_function_wave(lam, T)

    print("Nan values found for Bnu: " + str(Bnu[Bnu != Bnu]))
    print("Nan values found for Blam: " + str(Blam[Blam != Blam]))

    fig, ax = plt.subplots(1, 2, figsize=(15, 8), layout="tight")

    for i in range(len(T)):
        
        ax[0].plot(nu, Bnu[:, i], label=str(T[i]))
    
        ax[1].plot(lam, Blam[:, i])

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\nu~\mathrm{[H_z]}$", fontsize=fontsize)
    ax[0].set_ylabel(r"$B_\nu~\mathrm{[W.m^{-2}.st^{-1}.H_z^{-1}]}$", fontsize=fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize)
    ax[0].tick_params(axis='y', labelsize=fontsize)
    
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\lambda~\mathrm{[m]}$", fontsize=fontsize)
    ax[1].set_ylabel(r"$B_\lambda~\mathrm{[W.m^{-2}.st^{-1}.m^{-1}]}$", fontsize=fontsize)
    ax[1].tick_params(axis='x', labelsize=fontsize)
    ax[1].tick_params(axis='y', labelsize=fontsize)

    
    # ---
    # 2ND TEST: Several shapes for the temperature
    # ---

    N = 8

    T = 1000.
    T1 = np.full(N, T)
    T2 = np.full((N, N), T)
    T3 = np.full((N, N, N), T)

    fig, ax = plt.subplots(1, 2, figsize=(15, 8), layout="tight")

    ax[0].plot(nu, planck_function_freq(nu, T))
    ax[0].plot(nu, planck_function_freq(nu, T1)[:, np.random.randint(0, N-1, 1)])
    ax[0].plot(nu, planck_function_freq(nu, T2)[:, np.random.randint(0, N-1, 1), np.random.randint(0, N-1, 1)])
    ax[0].plot(nu, planck_function_freq(nu, T3)[:, np.random.randint(0, N-1, 1), np.random.randint(0, N-1, 1), np.random.randint(0, N-1, 1)])

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\nu~\mathrm{[H_z]}$", fontsize=fontsize)
    ax[0].set_ylabel(r"$B_\nu~\mathrm{[W.m^{-2}.st^{-1}.H_z^{-1}]}$", fontsize=fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize)
    ax[0].tick_params(axis='y', labelsize=fontsize)
    
    ax[1].plot(lam, planck_function_wave(lam, T))
    ax[1].plot(lam, planck_function_wave(lam, T1)[:, np.random.randint(0, N-1, 1)])
    ax[1].plot(lam, planck_function_wave(lam, T2)[:, np.random.randint(0, N-1, 1), np.random.randint(0, N-1, 1)])
    ax[1].plot(lam, planck_function_wave(lam, T3)[:, np.random.randint(0, N-1, 1), np.random.randint(0, N-1, 1), np.random.randint(0, N-1, 1)])

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\lambda~\mathrm{[m]}$", fontsize=fontsize)
    ax[1].set_ylabel(r"$B_\lambda~\mathrm{[W.m^{-2}.st^{-1}.m^{-1}]}$", fontsize=fontsize)
    ax[1].tick_params(axis='x', labelsize=fontsize)
    ax[1].tick_params(axis='y', labelsize=fontsize)

    plt.show()
