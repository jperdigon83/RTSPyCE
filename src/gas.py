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
import math as mt
import constants as ct
from scipy.interpolate import RegularGridInterpolator
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

table_wave = np.genfromtxt(dir_path + "/../data/gas/table_gaunt_wave.dat")
table_temp = np.genfromtxt(dir_path + "/../data/gas/table_gaunt_temp.dat")
n_wave, n_temp = len(table_wave), len(table_temp)
table_gaunt = np.genfromtxt(dir_path + "/../data/gas/table_gaunt.dat").reshape((n_temp, n_wave))

log_table_wave = np.log(table_wave)
log_table_temp = np.log(table_temp)
log_table_gaunt = np.log(table_gaunt)

# ---
# CONSTANTS
# ---

hkB = ct.h / ct.k_B

# ---
# FUNCTIONS
# ---

def gaunt_factor(wave, temp):
 
    if hasattr(temp, "__len__"):
        
        wave = wave.reshape([len(wave)] + len(temp.shape)*[1])
        temp = temp[None]
        dim = np.shape(wave)
        wave = wave * np.ones(temp.shape)
        temp = temp * np.ones(dim)

        idx = temp < np.finfo(float).tiny
        temp[idx] = np.finfo(float).tiny

        log_temp = np.log(temp)

    else:

        if temp < np.finfo(float).tiny:
            temp = np.finfo(float).tiny
        log_temp = mt.log(temp)
        
    log_wave = np.log(wave)
    
    log_gaunt_interp = RegularGridInterpolator((log_table_temp, log_table_wave), log_table_gaunt, bounds_error=False, fill_value=None)

    with np.errstate(over='ignore'):

        gf = np.exp(log_gaunt_interp((log_temp, log_wave)))
        
    return gf

def hydrogen_freefree_boundfree_absorption(freq, ne, temp):
    """ Computes the free-free and bound-free absorption cross sections for a gas of pure hydrogen
    """

    if hasattr(temp, "__len__") or hasattr(ne, "__len__"):

        assert temp.shape == ne.shape
    
    gf = gaunt_factor(ct.c/freq, temp)

    if hasattr(temp, "__len__"):
        
        freq = freq.reshape([len(freq)] + len(temp.shape)*[1])
        temp = temp[None]
        dim = np.shape(freq)
        freq = freq * np.ones(temp.shape)
        temp = temp * np.ones(dim)
        
    return 3.692349106159646e-2 * (1. - np.exp(-hkB*freq/temp)) * ne*ne * gf / (freq*freq*freq * np.sqrt(temp))
  
def thompson_scattering(ne):
    
    return ct.sigma_T * ne

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    N = 256

    fig, ax = plt.subplots(1, 2, figsize=(17, 8), layout="tight")
    fontsize = 20
    
    wave = np.logspace(-8, 0, N)
    temp = np.logspace(0, 9, 10)
    gff_bf = gaunt_factor(wave, temp)
    
    for i in range(len(temp)):
        ax[0].plot(wave, gff_bf[:, i], marker="+", label=r"$T = %.e ~ \mathrm{K}$" %temp[i])
        
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\lambda ~ \mathrm{[m]}$", fontsize=fontsize)
    ax[0].set_ylabel(r"$g_\mathrm{ff + bf}$", fontsize=fontsize)
    ax[0].legend(fontsize=fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize)
    ax[0].tick_params(axis='y', labelsize=fontsize)

    wave = np.logspace(-7, -2, 6)
    temp = np.logspace(0, 5, N)
    gff_bf = gaunt_factor(wave, temp)
    
    for i in range(len(wave)):
        ax[1].plot(temp, gff_bf[i], marker="+", label=r"$\lambda = %.2e ~ \mathrm{m}$"%wave[i])
        
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$T ~ \mathrm{[K]}$", fontsize=fontsize)
    ax[1].set_ylabel(r"$g_\mathrm{ff + bf}$", fontsize=fontsize)
    ax[1].legend(fontsize=fontsize)
    ax[1].tick_params(axis='x', labelsize=fontsize)
    ax[1].tick_params(axis='y', labelsize=fontsize)

    # Testing scalar values of the temperature
    
    wave = np.logspace(-8, 0, N)
    temp = 0.
    gff_bf = gaunt_factor(wave, temp)

    # Testing absorption coefficient

    freq = ct.c / wave
    temp = np.logspace(-5, 5, 6)
    ne_val = 1e-8
    ne = np.ones_like(temp)*ne_val

    print(_hydrogen_freefree_boundfree_absorption(ct.c/1e-6, ne_val, 1e3))

    kabs = hydrogen_freefree_boundfree_absorption(freq, ne, temp)

    plt.figure(figsize=(9, 8), layout="tight")
    
    for i in range(len(temp)):
        plt.plot(wave, kabs[:, i], label=r"$T = %.e ~ \mathrm{K}$" %temp[i])
        
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\lambda ~ \mathrm{[m]}$", fontsize=fontsize)
    plt.ylabel(r"$K_\lambda^\mathrm{abs}$", fontsize=fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title(r"$n_{e^-} = %.e \, \mathrm{kg.m^{-3}}$"%ne_val, fontsize=fontsize)
    
    plt.show()

    
    
    
    

    

    
    
