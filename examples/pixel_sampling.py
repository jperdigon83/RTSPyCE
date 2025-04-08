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
    # Initialisation of the RTSpyce class
    # ---
    
    foo = RTSpyce(r, theta)

    # ---
    # Cartesian image
    # ---

    n = 128
    n_half = int(0.5*n)
    
    L = 10*r_in
    dx = L/n
    incl = 60.  # in degrees
    
    x = np.linspace(0.5*dx, 0.5*(L-dx), n_half)  # We only consider x > 0
    y = np.linspace(-0.5*(L-dx), 0.5*(L-dx), n)
    x, y = np.meshgrid(x, y, indexing="ij")
    x = np.ravel(x)
    y = np.ravel(y)

    cartesian_map = foo.intensity_map(x, y, incl, S, Kext, I_star)
    cartesian_map = cartesian_map.reshape(nnu, n_half, n)
    cartesian_map = np.concatenate((cartesian_map[:, ::-1, :], cartesian_map), axis=1)

    # ---
    # Polar image
    # ---
    
    n = 128
    L = 5*r_in
    n_half = int(n/2)

    u_w_1 = np.linspace(0, r_star, 2, endpoint=False)
    u_w_2 = np.logspace(mt.log10(r_star), mt.log10(r_in), n_half-2, endpoint=False)
    u_w_3 = np.logspace(mt.log10(r_in), mt.log10(L), n_half+1)
    u_w = np.concatenate((u_w_1, u_w_2, u_w_3))
    u = 0.5 * (u_w[1:] + u_w[:-1])

    v_w = np.linspace(-0.5*mt.pi, 0.5*mt.pi, n_half+1) # We only consider x > 0
    v = 0.5 * (v_w[1:] + v_w[:-1])
    
    x = u[:, None] * np.cos(v[None, :])
    y = u[:, None] * np.sin(v[None, :])
    
    x = np.ravel(x)
    y = np.ravel(y)
    
    polar_map = foo.intensity_map(x, y, incl, S, Kext, I_star)
    polar_map = polar_map.reshape(nnu, n, n_half)
    polar_map = np.concatenate((polar_map, polar_map[:, :, ::-1]), axis=2)

    # ---
    # Elliptic image
    # ---
    
    u_out = mt.acosh(L/r_in)
    u_w = np.linspace(0., u_out, n+1)
    u = 0.5 * (u_w[1:] + u_w[:-1])
    
    v_w = np.linspace(-0.5*mt.pi, 0.5*mt.pi, n_half+1) # We only consider x > 0
    v = 0.5 * (v_w[1:] + v_w[:-1])

    x = r_in * np.cosh(u[:, None]) * np.cos(v[None, :])
    y = r_in * np.sinh(u[:, None]) * np.sin(v[None, :])
    
    x = np.ravel(x)
    y = np.ravel(y)

    elliptic_map = foo.intensity_map(x, y, incl, S, Kext, I_star)
    elliptic_map = elliptic_map.reshape(nnu, n, n_half)
    elliptic_map = np.concatenate((elliptic_map, elliptic_map[:, :, ::-1]), axis=2)

    # ---
    # Plotting results for lambda = 10 microns
    # ---
    
    fig, ax = plt.subplots(1, 3, figsize=(15., 5.), layout="constrained")
    
    fontsize = 20
    cmap = "afmhot"
    origin = "lower"
    aspect = "equal"
    vmin = 1e-30
    vmax = 1e-8
    
    fig.suptitle(r"$\lambda = {wave:.0e} \, m, \, \tau = {tau:.0e}$".format(wave=wave[1], tau=tau[1]), fontsize=fontsize)
    
    ax[0].set_title(r"$\text{Cartesian}$", fontsize=fontsize)	
    im1 = ax[0].imshow(cartesian_map[1].T, origin=origin, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax, clip=True), aspect=aspect)
    ax[0].set_xlabel(r"$x$", fontsize=fontsize)
    ax[0].set_ylabel(r"$y$", fontsize=fontsize)

    
    ax[1].set_title(r"$\text{Polar}$", fontsize=fontsize)	
    ax[1].imshow(polar_map[1].T, origin=origin, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax, clip=True), aspect=aspect)
    ax[1].set_xlabel(r"$u$", fontsize=fontsize)
    ax[1].set_ylabel(r"$v$", fontsize=fontsize)

    ax[2].set_title(r"$\text{Elliptic}$", fontsize=fontsize)	
    ax[2].imshow(elliptic_map[1].T, origin=origin, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax, clip=True), aspect=aspect)
    ax[2].set_xlabel(r"$u$", fontsize=fontsize)
    ax[2].set_ylabel(r"$v$", fontsize=fontsize)

    fig.colorbar(im1, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)

    plt.show()
