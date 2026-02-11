import sys
sys.path.insert(0, "../src/")
import source
import constants as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math as mt
import planck


if __name__ == "__main__":

    Rstar = 1.
    Tstar = 16500.
    loggstar = 4.5
    mhstar = 0.

    n = int(10000)
    numin = ct.c * 1e-2
    numax = ct.c * 1e9
    nu = np.logspace(mt.log10(numin), mt.log10(numax), n)
    wave = ct.c / nu
    
    src1 = source.AtlasATM("ck04models", Rstar, Tstar, loggstar, mhstar, wave)
    src2 = source.BlackBody(Rstar, Tstar, wave)
    
    integral_exact = ct.sigma_sb * Tstar**4 / mt.pi

    z = np.linspace(0., mt.log(numax/numin), n)

    integral_src1 = sp.integrate.trapezoid(src1.intensity*nu, x=z)
    integral_src2 = sp.integrate.trapezoid(src2.intensity*nu, x=z)

    err1 = np.abs(integral_src1 - integral_exact) / integral_exact
    err2 = np.abs(integral_src2 - integral_exact) / integral_exact

    print(err1, err2)
    
    fig = plt.figure()

    plt.plot(wave, src1.intensity, label="CK04")
    plt.plot(wave, src2.intensity, label="BB")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
