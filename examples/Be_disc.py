import sys
sys.path.insert(0, "../src/")

import numpy as np
import math as mt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import image
import planck
import constants as ct
import envelope
import source

if __name__ == "__main__":

    L = 10
    N = 128
    incl = 30.
    PA = 0.
    d = 1e8
    
    wave_J = np.linspace(1.1, 1.3, 9)
    wave_H = np.linspace(1.5, 1.8, 10)
    wave_K = np.linspace(2.0, 2.4, 13)
    wave_L = np.linspace(3.2, 3.6, 13)
    
    wave_M = np.linspace(4.5, 4.9, 9)
    wave_N = np.linspace(9.0, 12.0, 13)
    wave_Q = np.linspace(18.0, 24.0, 13)

    wave = np.concatenate((wave_J, wave_H, wave_K, wave_L, wave_M, wave_N, wave_Q))
    wave = np.unique(wave)
    wave = np.sort(wave)
    wave *= 1e-6

    #wave = np.logspace(mt.log10(1.1), mt.log10(24.0), 64) * 1e-6
    
    nwave = len(wave)
    nu = ct.c / wave
    img = image.UniformCartesianImage(N, L, incl, PA, d, wave)

    Rstar = 1.0
    Tstar = 16500.
    src = source.BlackBody(Rstar, Tstar, wave)
    
    nr, ntheta = 128, 128
    r = np.logspace(mt.log10(Rstar), mt.log10(50*Rstar), nr)
    theta = np.linspace(0., 0.5*mt.pi, ntheta)
    theta = np.arccos(np.linspace(1., 0., ntheta))

    R = r[:, None] * np.sin(theta[None, :])
    Z = r[:, None] * np.cos(theta[None, :])
    
    tau = 1e2
    H0 = 0.1
    K0 = tau / (mt.sqrt(2.*mt.pi) * H0) 

    H = H0 * (R/Rstar)**(3./2)
    Kext = K0 * (R/Rstar)**(-7./2) * np.exp(-0.5 * (Z/H)**2)
    Kext[:, 0] = Kext[:, 1]
    Kext = np.repeat(Kext[None, :, :], nwave, axis=0)

    T = Tstar * (R / Rstar)**(-3./4)
    T[:, 0] = T[:, 1]
    S = planck.planck_function_freq(nu, T)
    
    env = envelope.Envelope(wave, r, theta, Kext, S)

    img.compute_intensity(env, src)
    flux = img.compute_flux()

    # Generate a random array of telescopes

    N_tel = 4

    x_tel = np.array([-20., 0., 40, 0.])
    y_tel = np.array([-20., 0., -40, 80.])
    # x_tel = np.random.uniform(-100., 100., N_tel)
    # y_tel = np.random.uniform(-100., 100., N_tel)

    u = []
    v = []
    B = []
    
    for i in range(N_tel):
        for j in range(i+1, N_tel):

            ut = x_tel[j] - x_tel[i]
            vt = y_tel[j] - y_tel[i]
            
            u.append(ut)
            v.append(vt)
            
            B.append(mt.sqrt(ut*ut + vt*vt))
            
    u = np.array(u)
    v = np.array(v)
    B = np.array(B)

    # Computing Fourier Transform and Visibilities
    
    ft = img.compute_fourier_transform(u, v)
    V2 = (np.abs(ft) / flux)**2
    
    intensity = img.reconstruct_image()
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), layout="tight")
    
    for a in ax.flat:
        a.set_box_aspect(1)
    
    alphaw = 206264806.2471 * img.xw/d 
    betaw = 206264806.2471 * img.yw/d

    #c = ax[0, 0].pcolormesh(alpha_mas, beta_mas, intensity[0], cmap="afmhot")
    c = ax[0, 0].pcolormesh(alphaw, betaw, intensity[0], cmap="afmhot", norm=LogNorm(clip=True))
    cbar = fig.colorbar(c, ax=ax[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
    ax[0, 0].set_xlabel(r"$\alpha ~ [mas]$")
    ax[0, 0].set_ylabel(r"$\beta ~ [mas]$")

    ax[0, 1].scatter(x_tel, y_tel, c="k", marker="+")
    ax[0, 1].set_xlim(-100., 100)
    ax[0, 1].set_ylim(-100., 100)
    ax[0, 1].set_xlabel(r"$x ~ [m]$")
    ax[0, 1].set_ylabel(r"$y ~ [m]$")

    colors = np.array(["tab:orange", "tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"])

    for i in range(len(B)):

        ulmas = u[i] / (wave*206264806.2471)
        vlmas = v[i] / (wave*206264806.2471)
        Blmas = B[i] / (wave*206264806.2471)
    
        ax[1, 0].scatter(ulmas, vlmas, c=colors[i], marker=".")
        ax[1, 0].scatter(-ulmas, -vlmas, c=colors[i], marker=".")
        ax[1, 1].scatter(Blmas, V2[i, :], c=colors[i], marker="+")
        

    ax[1, 0].set_xlabel(r"$u / \lambda ~ [mas^{-1}]$")
    ax[1, 0].set_ylabel(r"$v / \lambda ~ [mas^{-1}]$")
    
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel(r"$B / \lambda ~ [mas^{-1}]$")
    ax[1, 1].set_ylabel(r"$|V|^2$")

    plt.savefig("Be_disc.pdf")
