import os
pathdir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append(pathdir+'/../src')
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import rtspyce as RTI

plt.close('all')

if __name__=="__main__":
    
    """
    This script is an example file
    """

    Rstar = 1.
    Rin = 3.
    Rout = 6.
    tau = 10.
    
    nr = 128
    ntheta = 128
    incl = np.deg2rad(60.)

    theta = np.linspace(0., 0.5*mt.pi, ntheta)

    rCav = np.linspace(Rstar, Rin-1e-5, 2)
    rEnv = np.linspace(Rin, Rout, nr)
    r = np.concatenate((rCav, rEnv))

    KextCav = np.zeros((1, 2, ntheta))

    zEnv = rEnv[:, None] * np.cos(theta[None, :])
    K0Env = tau / (Rout - Rin)
    HEnv = 0.2
    KextEnv = np.ones((1, nr, ntheta)) * K0Env * np.exp(-0.5*(zEnv/HEnv)**2)
    
    Kext = np.concatenate((KextCav, KextEnv), axis=1)

    SCav = np.zeros((1, 2, ntheta))
    SEnv = np.ones((1, nr, ntheta))
    S = np.concatenate((SCav, SEnv), axis=1)

    Istar = np.ones(1)

    L = 3. * Rout

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), layout='tight')

    N = 256
    fontsize = 20
    aspect = "auto"
    L = 2.5 * Rout
    interpolation = 'none'
    levels = np.logspace(-3, mt.log10(0.99), 8)
    linewidths = 0.5
    colors = "w"

    #
    # CARTESIAN IMAGE
    #
    
    dx = L / N

    x = np.linspace(-0.5*L+0.5*dx, 0.5*L-0.5*dx, N)
    y = np.copy(x)

    x, y = np.meshgrid(x, y, indexing='ij')

    model = RTI.RTSPyCE(np.ravel(x), np.ravel(y), r, theta, Kext, S, Istar, incl=incl)

    imag = np.reshape(model.intensityMap, (1, N, N))

    extent = [-0.5*L, 0.5*L, -0.5*L, 0.5*L]
    
    ax[0].imshow(np.transpose(imag[0]), origin='lower', extent=extent, aspect=aspect, interpolation=interpolation)
    # ax[0].contour(x, y, imag[0], levels=levels, linewidths=linewidths, colors=colors)
    ax[0].set_xlabel(r"$x_\mathrm{I}~[R_\mathrm{\star}]$", fontsize=fontsize)
    ax[0].set_ylabel(r"$y_\mathrm{I}~[R_\mathrm{\star}]$", fontsize=fontsize)
    ax[0].tick_params(labelsize="large")
   
    #
    # SPHERICAL IMAGE
    #

    rI = np.linspace(0, 0.5*L, N+1)
    rI = 0.5*(rI[1:] + rI[:-1])

    thetaI = np.linspace(0., 2*mt.pi, N+1)
    thetaI = 0.5*(thetaI[1:] + thetaI[:-1])

    xI = rI[:, None] * np.cos(thetaI[None, :])
    yI = rI[:, None] * np.sin(thetaI[None, :])

    model = RTI.RayTracingImage(np.ravel(xI), np.ravel(yI), r, theta, Kext, S, Istar, rStar=Rstar, incl=incl)

    imag = np.reshape(model.intensityMap, (1, N, N))

    extent = [0, rI[-1], 0, 360]
   
    ax[1].imshow(np.transpose(imag[0]), origin='lower', extent=extent, aspect=aspect, interpolation=interpolation)
    # ax[1].contour(rI, thetaI, np.transpose(imag[0]), levels=levels, linewidths=linewidths, colors=colors)
    ax[1].set_xlabel(r"$r_\mathrm{I}~[R_\mathrm{\star}]$", fontsize=fontsize)
    ax[1].set_ylabel(r"$\theta_\mathrm{I}~[\mathrm{deg}]$", fontsize=fontsize)
    ax[1].tick_params(labelsize="large")
    #
    # ELLIPTIC IMAGE
    #

    N = 256
    uI = np.linspace(0, mt.acosh(0.5*L/Rin), N+1)
    uI = 0.5*(uI[1:] + uI[:-1])

    vI = np.linspace(0., 2*mt.pi, N+1)
    vI = 0.5*(vI[1:] + vI[:-1])

    xI = Rin * np.cosh(uI[:, None]) * np.cos(vI[None, :])
    yI = Rin * np.sinh(uI[:, None]) * np.sin(vI[None, :])

    model = RTI.RayTracingImage(np.ravel(xI), np.ravel(yI), r, theta, Kext, S, Istar, rStar=Rstar, incl=incl)

    imag = np.reshape(model.intensityMap, (1, N, N))

    extent = [0, uI[-1], 0, 360]
   
    ax[2].imshow(np.transpose(imag[0]), origin='lower', extent=extent, aspect=aspect, interpolation=interpolation)
    # ax[2].contour(uI, vI, np.transpose(imag[0]), levels=levels, linewidths=linewidths, colors=colors)
    ax[2].set_xlabel(r"$\mu_\mathrm{I}~[R_\mathrm{\star}]$", fontsize=fontsize)
    ax[2].set_ylabel(r"$\nu_\mathrm{I}~[\mathrm{deg}]$", fontsize=fontsize)
    ax[2].tick_params(labelsize="large")
  
    plt.show()

    

    

    
