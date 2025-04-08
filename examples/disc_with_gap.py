import numpy as np
import math as mt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, parent_dir)

from rtspyce import RTSpyce
from planck import planck_function_freq
import constants as ct

if __name__ == "__main__":

    nr1 = 128
    nr2 = 128
    ntheta = 128
    n = 512

    T_star = 5750.
    M_star = 1. * ct.M_sun
    
    r_star = 1.
    r1_in = 0.2 * ct.au / ct.R_sun
    r1_out = 0.9 * ct.au / ct.R_sun
    r2_in = 1.1 * ct.au / ct.R_sun
    r2_out = 3. * ct.au / ct.R_sun

    wave = np.array([1e-6, 1e-5, 1e-4, 5e-4, 1e-3])
    nnu = len(wave)
    nu = ct.c / wave

    r_cavity = np.linspace(r_star, r1_in - 1e-5, 2)
    r1 = np.logspace(mt.log10(r1_in), mt.log10(r1_out), nr1)
    r_gap = np.linspace(r1_out + 1e-5, r2_in - 1e-5, 2)
    r2 = np.logspace(mt.log10(r2_in), mt.log10(r2_out), nr2)

    r = np.concatenate((r_cavity, r1, r_gap, r2))
    nr = len(r)

    theta = np.arccos(np.linspace(1., 0, ntheta))

    # ---
    # Inner cavity
    # ---

    T_cavity = np.zeros((2, ntheta))
    S_cavity = np.zeros((nnu, 2, ntheta))
    Kext_cavity = np.zeros_like(S_cavity)

    # ---
    # Fist disc 
    # ---

    tau1 = np.logspace(3, -1, nnu)
    alpha1 = -0.75
    beta1 = -1.
    T1_in = T_star * mt.sqrt(r_star/r1_in)
  
    R1 = r1[:, None] * np.sin(theta[None, :])
    z1 = r1[:, None] * np.cos(theta[None, :])

    T1 = np.empty((nr1, ntheta))
    T1[:, 1:] = T1_in * (R1[:, 1:]/r1_in)**alpha1
    T1[:, 0] = T1[:, 1]

    S1 = planck_function_freq(nu, T1)
    
    if beta1 != -1.:

        pow_K = beta1 + 1.
        K_in = pow_K * tau1 / (r1_in * ((r1_out/r1_in)**pow_K - 1.))
	
    else:
        
        K_in = tau1 / (r1_in * mt.log(r1_out/r1_in))
		
    H1 = np.sqrt(ct.k_B * T1 * R1**3 * ct.R_sun / (2. * ct.m_p * ct.G * M_star))
	
    Kext1 = np.empty((nnu, nr1, ntheta))
    
    Kext1[:, :, 1:] = K_in[:, None, None] * (R1[None, :, 1:]/r1_in)**beta1 * np.exp(-0.5 * (z1[None, :, 1:]/H1[None, :, 1:])**2)
    Kext1[:, :, 0] = Kext1[:, :, 1]  # For the polar axis

    # ---
    # Second disc 
    # ---

    tau2 = np.logspace(3, -1, nnu)
    
    T2_in = T_star * mt.sqrt(r_star/r2_in)
    alpha2 = -0.75
    beta2 = -1.
   
    R2 = r2[:, None] * np.sin(theta[None, :])
    z2 = r2[:, None] * np.cos(theta[None, :])
    
    T2 = np.empty((nr2, ntheta))
    T2[:, 1:] = T2_in * (R2[:, 1:]/r2_in)**alpha2
    T2[:, 0] = T2[:, 1]
    S2 = planck_function_freq(nu, T2)
    
    if beta2 != -1.:

        pow_K = beta2 + 1.
        K_in = pow_K * tau2 / (r1_in * ((r2_out/r2_in)**pow_K - 1.))
		
    else:
        
        K_in = tau2 / (r2_in * mt.log(r2_out/r2_in))
		
    H2 = np.sqrt(ct.k_B * T2 * R2**3 * ct.R_sun / (2. * ct.m_p * ct.G * M_star))
	
    Kext2 = np.empty((nnu, nr2, ntheta))
    
    Kext2[:, :, 1:] = K_in[:, None, None] * (R2[None, :, 1:]/r2_in)**beta2 * np.exp(-0.5 * (z2[None, :, 1:]/H2[None, :, 1:])**2)
    Kext2[:, :, 0] = Kext2[:, :, 1]  # For the polar axis

    # ---
    # Concatenate discs and gaps
    # ---

    T = np.concatenate((T_cavity, T1, T_cavity, T2), axis=0)
    S = np.concatenate((S_cavity, S1, S_cavity, S2), axis=1)
    Kext = np.concatenate((Kext_cavity, Kext1, Kext_cavity, Kext2), axis=1)

    # ---
    # Stellar radiation
    # ---
    
    I_star = planck_function_freq(nu, T_star)
    
    # ---
    # Image
    # ---

    n_half = int(0.5*n)
    
    L = 2.1 * r2_out
    dx = L/n
    incl = 50.  # in degrees
    
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
    
    # fig, ax = plt.subplots(1, nnu, figsize=(nnu*4., 4.), layout="constrained")
    
    fontsize = 20
    xlim = 0.5*L*ct.R_sun / ct.au
    extent = (-xlim, xlim, -xlim, xlim)
    cmap = "afmhot"

    fig, ax = plt.subplots(2, 3, figsize=(15, 10), layout="tight")


    ax[1, 2].plot(r*ct.R_sun/ct.au, Kext[0, :, -1]/np.max(Kext[0, :, -1]))
    ax[1, 2].set_yscale("log")

    ax[1, 2].set_xlabel(r"$R \, [\mathrm{AU}]$", fontsize=fontsize)
    ax[1, 2].set_ylabel(r"$\frac{\Sigma_\mathrm{dust}}{\Sigma_0}$", fontsize=fontsize)
    
    idx = 0
    for ax in ax.flat[:-1]:
        
        ax.set_title(r"$\lambda = {wave:.0e} \, \mathrm{{m}}$".format(wave=wave[idx]), fontsize=fontsize)
		
        ax.imshow(intensity_map[idx].T, origin="lower", extent=extent, cmap=cmap, interpolation="antialiased")
    
        idx += 1

    fig.supxlabel(r"$x \, \mathrm{[AU]}$", fontsize=fontsize)
    fig.supylabel(r"$y \, \mathrm{[AU]}$", fontsize=fontsize)
    plt.show()
