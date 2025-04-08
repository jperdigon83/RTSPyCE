import numpy as np
import math as mt
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, parent_dir)

from rtspyce import RTSpyce
from planck import planck_function_freq
import constants as ct

if __name__ == "__main__":

    # ---
    # The CS grid
    # ---

    r_star = 1.  # In solar radii
    r_in = 100.
    r_out = 1000. * r_in

    nr, ntheta = 128, 128

    r_cavity = np.linspace(r_star, r_in - 1e-5, 2)
    r_disc = np.logspace(mt.log10(r_in), mt.log10(r_out), nr)
    r = np.concatenate((r_cavity, r_disc))
    
    theta = np.arccos(np.linspace(1., 0., ntheta))

    # ---
    # Source function
    # ---
    
    nnu = 4
    wave = np.logspace(-6, -3, nnu)  # From 1 micron to 1000 microns 
    nu = ct.c / wave

    R = r[:, None] * np.sin(theta[None, :])  # Equatorial radius
    z = r[:, None] * np.cos(theta[None, :])  # Vertical coordinate

    T_in = 1300.  # Dust inner temperature
    alpha = -0.5  # radial exponent of the temperature

    T = np.empty_like(R)
    T[:, 1:] = T_in * (R[:, 1:]/r_star)**alpha
    T[:, 0] = T[:, 1]  # For the polar axis (R/r_star diverges at the pole for alpha < 0)
    T[:2, :] = 0.  # For the inner cavity

    S = planck_function_freq(nu, T)

    # ---
    # Extinction coefficient
    # ---
    
    tau = np.logspace(4., 1., nnu)  # from 10 000 at 1 microns to 10 at 1000 microns
    beta = -3.5
    gamma = 1.3
    H_in = 5.  # in solar radii
	
    if beta != -1.:

        pow_K = beta + 1.
        K_in = pow_K * tau / (r_in * ((r_out/r_in)**pow_K - 1.))
		
    else:
        
        K_in = tau / (r_in * mt.log(r_out/r_in))
		
    H = H_in * (R/r_in)**gamma
	
    Kext = np.empty_like(S)
    Kext[:, :, 1:] = K_in[:, None, None] * (R[None, :, 1:]/r_in)**beta * np.exp(-0.5 * (z[None, :, 1:]/H[None, :, 1:])**2)
    Kext[:, :, 0] = Kext[:, :, 1]  # For the polar axis
    Kext[:, :2, :] = 0.  # For the inner cavity

    # ---
    # Stellar radiation
    # ---
    
    T_star = 5750.
    I_star = planck_function_freq(nu, T_star)

    # ---
    # Image
    # ---

    n = 128
    n_half = int(0.5*n)
	
    L = 3 * r_in
    dx = L/n
    incl = 60.  # in degrees
    
    x = np.linspace(0.5*dx, 0.5*(L-dx), n_half)  # We only consider x > 0
    y = np.linspace(-0.5*(L-dx), 0.5*(L-dx), n)
    x, y = np.meshgrid(x, y, indexing="ij")
    x = np.ravel(x)
    y = np.ravel(y)
    
    # ---
    # RTSpyce
    # ---
    
    foo = RTSpyce(r, theta)

    intensity_map = foo.intensity_map(x, y, incl, S, Kext, I_star)
    intensity_map = intensity_map.reshape(nnu, n_half, n)
    intensity_map = np.concatenate((intensity_map[:, ::-1, :], intensity_map), axis=1)

    # ---
    # Plotting results
    # ---
    
    fig, ax = plt.subplots(1, nnu, figsize=(nnu*4., 4.), layout="constrained")
    
    fontsize = 20
    extent = (-0.5*L, 0.5*L, -0.5*L, 0.5*L)
    cmap = "afmhot"

   
    for k in range(nnu):

        ax[k].set_title(r"$\lambda = {wave:.0e} \, m, \, \tau = {tau:.0e}$".format(wave=wave[k], tau=tau[k]))
		
        ax[k].imshow(intensity_map[k].T, origin="lower", extent=extent, cmap=cmap)

    fig.supxlabel(r"$x \, \mathrm{[R_\odot]}$", fontsize=fontsize)
    fig.supylabel(r"$y \, \mathrm{[R_\odot]}$", fontsize=fontsize)

    plt.show()
