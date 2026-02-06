import os
pathdir = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(pathdir+'/../matter')
sys.path.append(pathdir+'/../matter/gas')
sys.path.append(pathdir+'/../src')

import matplotlib.pyplot as plt
import numpy as np
import math as mt
import planck as pl

import RayTracingImage as RTI
from astropy import constants as const
import optical_coeff as optcoeff
import time

plt.close('all')

if __name__=="__main__":
    
    """
    This script is an example file
    """

    # Parameters
   
    # Stellar parameters
    
    Tstar = 16500.  # [K]
    Rstar = 3.4*const.R_sun.value  # [m]
    estar = 0.  # Stellar excentricity e = sqrt(1. - (Rpole/Requator)**2): 0 <= e <= 1
    Mstar = 6.7*const.M_sun.value  # [kg]
    
    # Disc parameters
    
    Tin = 0.6*Tstar  # [K]
    Rout = 50*Rstar  # [m] Outer radius: rout > rin
    ionisation_factor = 1.  # 0 <= ionisation_factor <= 1
    rho0 = 1e-8  # [kg/m^[3]
    alpha = -3.5  # exponent of radial density law
    beta = 1.5  # exponent of scale height parameter
    gamma = 0.  # expondent of the radial temperature law
    
    # Image parameters
    
    N = 128  # N must be even
    L = 10*Rstar  # [m] Image size (carthesian square image)
    incl = np.deg2rad(60.)  # 0 <= incl <= 0.5*np.pi
    d = 1.*const.pc.value
    
    nr = 128
    ntheta = 64
    nlam = 8
    
    # Wavelength parameters

    lam = np.logspace(-7, -2, nlam)
    nlam = len(lam)
    nu = const.c.value / lam

    r = Rstar*np.exp(mt.log(Rout/Rstar)*np.linspace(0., 1., nr))

    m = 5.
    theta = np.tan(np.linspace(0., mt.atan(0.5*mt.pi)**m, ntheta)**(1./m))

    R = r[:, None]*np.sin(theta[None, :])
    Z = r[:, None]*np.cos(theta[None, :])
    
    T = np.empty((nr, ntheta))
    T[:, 1:] = Tin*(R[:, 1:]/Rstar)**gamma
    T[:, 0] = T[:, 1]
  
    mu = 1./(1. + ionisation_factor)
    
    H0 = np.sqrt(const.k_B.value*Rstar**3*T/(mu*const.m_p.value*const.G.value*Mstar))
    
    H = H0 * (R/Rstar)**beta

    rho = np.empty_like(H)
    rho[:, 1:] = rho0 * (R[:, 1:]/Rstar)**alpha * np.exp(-0.5*(Z[:, 1:]/H[:, 1:])**2)
    rho[:, 0] = 0.
    
    Kext = optcoeff.freefree_boundfree_absorption_coeff(nu, rho/(mu*const.m_p.value), T, ionisation_factor)
    
    S = pl.planckFunction(nu, T)

    Istar = pl.planckFunction(nu, Tstar)

    # Building a cube of carthesian image
    
    dx = L/N
    x = np.linspace(0.5*(dx-L), 0.5*(L-dx), N)
    y = np.linspace(0.5*(dx-L), 0.5*(L-dx), N)
    x, y = np.meshgrid(x, y, indexing='ij')

    t0 = time.time()
    
    model = RTI.RayTracingImage(np.ravel(x), np.ravel(y), r, theta, Kext, S, Istar, rStar=Rstar, eStar=estar, incl=incl)

    imag = np.reshape(model.intensityMap, (nlam, N, N))

    t1 = time.time()

    print(str(t1-t0))

    for m in range(nlam):
        
        plt.figure()
        plt.imshow(np.transpose(imag[m]), origin='lower')
        plt.tight_layout()
        
    # Building a cube of spherical images

    # uw = np.linspace(0, Rstar, 4)
    # uw = np.concatenate((uw, np.logspace(mt.log10(Rstar),mt.log10(Rout), 32)))
    # uw = np.unique(uw)
    # u = 0.5*(uw[1:] + uw[:-1])
    # du = uw[1:] - uw[:-1]
    # nu = len(u)
    
    # vw = np.linspace(0, 2*mt.pi, 32)
    # v = 0.5*(vw[1:] + vw[:-1])
    # dv = vw[1:] - vw[:-1]
    # nv = len(v)

    # u, v = np.meshgrid(u, v, indexing='ij')

    # x = u*np.sin(v)
    # y = u*np.cos(v)

    # t0 = time.time()

    # model = RTI.RayTracingImage(np.ravel(x), np.ravel(y), r, theta, rStar=Rstar, eStar=estar, incl=incl)
    
    # model.compute_image(Kext, S, Istar)

    # imag = np.reshape(model.intensityMap, (nlam, nu, nv))

    # t1 = time.time()

    # print(str(t1 - t0))
    
    # for m in range(nlam):
        
    #     plt.figure()
    #     plt.imshow(np.transpose(imag[m]), origin='lower')
    #     plt.tight_layout()

    # ds = u * du[:, None] * dv[None, :] / d**2

    # Computing the sed of the model
    
    # flux = np.sum(imag*ds, axis=(1, 2))
    # sed = nu*flux

    # flux_star = mt.pi*Istar*(Rstar/d)**2
    # sed_star = nu*flux_star

    # plt.figure()
    # plt.scatter(lam, sed, marker='+', color='r')
    # plt.plot(lam, sed_star, 'k--')

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.tight_layout()

    plt.show()

    

    

    
