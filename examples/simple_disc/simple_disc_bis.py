import numpy as np
import math as mt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "../../src"))
sys.path.insert(0, parent_dir)

from rtspyce import RTSPyCE
from planck import planck_function_freq
import constants as ct
from envelope import Envelope
from source import BlackBody
from image import UniformCartesianImage, PolarImage


class Disc(Envelope):

    def __init__(self, params):

        rin = params["rin"]
        rout = params["rout"]
        nr = params["nr"]
        ntheta = params["ntheta"]

        Tin = params["Tin"]
        powT = params["powT"]

        wave = params["wave"]
        tau = params["tau"]
        pown = params["pown"]
        Hin = params["Hin"]
        powH = params["powH"]

        # ---
        # The CS grid
        # ---
        
        r_cavity = np.linspace(1., rin - 1e-5, 2)
        r_disc = np.logspace(mt.log10(rin), mt.log10(rout), nr)
        r = np.concatenate((r_cavity, r_disc))

        theta = np.arccos(np.linspace(1., 0., ntheta))

        # Cylindrical coordinates
        
        R = r[:, None] * np.sin(theta[None, :])  
        z = r[:, None] * np.cos(theta[None, :])

        # ---
        # The Source function
        # ---
        
        T = np.empty_like(R)
        T[:, 1:] = Tin * (R[:, 1:])**powT
        T[:, 0] = T[:, 1]  # For the polar axis (R/r_star diverges at the pole for alpha < 0)
        T[:2, :] = 0.  # For the inner cavity

        nu = ct.c / wave
        S = planck_function_freq(nu, T)

        # ---
        # Extinction coefficient
        # ---

        Kin = tau / (Hin * mt.sqrt(2*mt.pi))
       
		
        H = Hin * (R/rin)**powH
	
        Kext = np.empty_like(S)
        Kext[:, :, 1:] = Kin[:, None, None] * (R[None, :, 1:]/rin)**pown * np.exp(-0.5 * (z[None, :, 1:]/H[None, :, 1:])**2)
        Kext[:, :, 0] = Kext[:, :, 1]  # For the polar axis
        Kext[:, :2, :] = 0.  # For the inner cavity


        super().__init__(wave, r, theta, Kext, S)
    


if __name__ == "__main__":

    wave = np.array([10., 20., 100., 500]) * 1e-6
    
    params = {
        
        "rin": 200.,          # Example inner radius
        "rout": 1e5,        # Example outer radius
        "nr": 128,           # Example number of radial grid points
        "ntheta": 64,        # Example number of angular grid points
        
        "Tin": 1500.,         # Example temperature at inner boundary
        "powT": -0.75,         # Example temperature power law exponent
        
        "wave": wave, # Example wavelength
        "tau":  np.logspace(3, 1, 4), # Example optical depth
        "pown": -3.5,         # Example power law exponent for number density
        "Hin": 5,         # Example parameter related to hydrogen density
        "powH": 1.5         # Example power law exponent for hydrogen density
    }

    env = Disc(params)

    Rstar = 1.
    Tstar = 9500.
    src = BlackBody(Rstar, Tstar, wave)

    # Carthesian image

    N = 128
    L = 500.
    incl = 45.
    PA = 0.
    d = 5e7
    
    img = UniformCartesianImage(N, L, incl, PA, d, wave)
    img.compute_intensity(env, src)
    #img.add_star(env, src)
    intensity = img.reconstruct_image()

    # Polar image
    
    R = 400.
    nu = 128
    nv = 128
   
    img2 = PolarImage(Rstar, 1, R, nu, nv, incl, PA, d, wave)
    img2.compute_intensity(env, src)
    
    intensity2 = img2.reconstruct_image()


    fig, ax = plt.subplots(2, 4, figsize=(16, 8), layout="tight")

    for i in range(4):

        c = ax[0, i].pcolormesh(img.xw, img.yw, intensity[i], cmap="afmhot")
       
        cbar = fig.colorbar(c, ax=ax[0, i], orientation='vertical', fraction=0.046, pad=0.04)

        
        c = ax[1, i].pcolormesh(img2.vw, img2.uw, intensity2[i], cmap="afmhot")
        cbar = fig.colorbar(c, ax=ax[1, i], orientation='vertical', fraction=0.046, pad=0.04)

        ax[0, i].set_aspect('equal')
        ax[1, i].set_yscale("symlog")
        ax[1, i].set_aspect('equal')
        
    ax[0, 1].set_xlabel(r"$x \, \mathrm{[R_\odot]}$")
    ax[0, 0].set_ylabel(r"$y \, \mathrm{[R_\odot]}$")

    ax[1, 1].set_xlabel(r"$v \, \mathrm{[rad]}$")
    ax[1, 0].set_ylabel(r"$u \, \mathrm{[R_\odot]}$")
  
    
  
   
    
    
    plt.show()


    

    

    




    
