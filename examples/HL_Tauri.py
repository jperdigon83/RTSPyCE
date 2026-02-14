import sys
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm

sys.path.insert(0, "../src/")

from envelope import Envelope
import constants as ct
from planck import planck_function_freq
from source import BlackBody
import image
from dust import opacities

class Disc(Envelope):

    def __init__(self, params):

        # Wavelengths and frequencies vectors
        
        nu = ct.c / params["wave"]

        # Circumstellar grid
        
        r_cavity = np.linspace(params["rstar"], params["rin"] - 1e-3, 2)
        r_disc = np.logspace(mt.log10(params["rin"]), mt.log10(params["rout"]), params["nr"]-2)
        r = np.concatenate((r_cavity, r_disc))

        theta = np.acos(np.linspace(1., 0., params["ntheta"]))

        R = r[:, None] * np.sin(theta[None, :])  # Cylindrical coordinates
        z = r[:, None] * np.cos(theta[None, :])  #       

        R_rin = R / params["rin"]
        
        # Temperature profile and Source function
        
        T = np.empty_like(R)
        T[:, 1:] = params["Tin"] * R_rin[:, 1:]**params["powT"]
        T[:, 0] = T[:, 1]  # For the polar axis (R/r_star diverges at the pole for alpha < 0)
        T[:2, :] = 0.  # For the inner cavity
        
        # Thermal emission
        
        S = planck_function_freq(nu, T)

        # Opacities of dust
        
        a = np.logspace(mt.log10(params["amin"]), mt.log10(params["amax"]), params["na"])
        
        fa = a**params["powa"]

        refidx = np.genfromtxt("../data/dust/sil-draine.nk")
        
        Cabs, Csca = opacities(params["wave"], [refidx], [a], [fa], [1.])
        Cext = Cabs + Csca

        Cabs0, Csca0 = opacities(np.array([params["wave0"]]), [refidx], [a], [fa], [1.])
        Cext0 = Cabs0 + Csca0
        
        # number density of dust grains

        Hin = params["coeff_Hin"] * mt.sqrt(ct.k_B * params["Tin"] * ct.R_sun * params["rin"]**3 / (2.3 * 1.66e-27 * ct.G * ct.M_sun * params["Mstar"]))

        H =  Hin * R_rin**params["powH"]

        nin = params["tau0"] / (mt.sqrt(2*mt.pi) * Cext0 * Hin)

        n = np.empty_like(R)
        n[:, 1:] = nin * R_rin[:, 1:]**params["pown"] * np.exp(-0.5 * (z[:, 1:]/H[:, 1:])**2)
        n[:, 0] = n[:, 1]
        n[:2, :] = 0.

        Kext = Cext[:, None, None] * n[None, :, :]

        super().__init__(params["wave"], r, theta, Kext, S)

if __name__ == "__main__":


    sun2au = 4.6524726e-3
    au2sun = 1. / sun2au
    ly2sun = 13593003.553
    mas2rad = 4.8481e-6
    
    params = {

        "rstar": 1.6,
        "Mstar": 1.7,
        "rin": 0.1 * au2sun,
        "rout": 150. * au2sun,
        "nr": 128,
        "ntheta": 100,
        "Tin": 1500.,
        "powT": -1./2,
        "wave":  np.array([0.8, 1., 2.5]) * 1e-3,
        "amin": 1e-8,
        "amax": 1e-4,
        "na": 32,
        "powa": -3.5,
        "coeff_Hin": 1.,
        "powH": 5./4,
        "pown": -11./4,
        "tau0": 100.,
        "wave0": 0.55e-6,
    }

    
    env = Disc(params)

    Tstar = 4800.
    
    src = BlackBody(params["rstar"], Tstar, params["wave"]) 

    N = 256
    incl = 60.
    PA = 0.
    d = 450 * ly2sun
    L = d * 2 * mas2rad
    
    rec_img = image.UniformCartesianImage(N, L, incl, PA, d, params["wave"]) 
    rec_img.compute_intensity(env, src)
    imap = rec_img.reconstruct_image()

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    i = 0

    vmin = 1e-30
    vmax = np.max(imap)
    
    for ax in axes.flat:
        
        im = ax.pcolormesh(rec_img.xw, rec_img.yw, imap[i], cmap="afmhot", norm=LogNorm(clip=True))
        # im = ax.pcolormesh(rec_img.xw, rec_img.yw, imap[i], cmap="afmhot")
        # cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        i += 1


    # params["wave"] = np.logspace(-7, -3, 128)
    # env = Disc(params)

    # src = BlackBody(params["rstar"], Tstar, params["wave"])

    # incl = np.linspace(0., 90., 6)


    # plt.figure()

    # nu = ct.c/env.wave
    
    # for i in range(len(incl)):

    #     polar_img = image.PolarImage(params["rstar"], 2, params["rout"], 64, 64, incl[i], PA, d, params["wave"])

    #     polar_img.compute_intensity(env, src)

    #     flux = polar_img.compute_flux()

    #     plt.plot(polar_img.wave, flux*nu)
    

    # Fstar = mt.pi*src.intensity*(src.R/polar_img.d)**2

    # plt.plot(polar_img.wave, Fstar*nu, "k--")
    # plt.xscale("log")
    # plt.yscale("log")

    
    
    
    plt.show()
    
    

    
