import sys
sys.path.insert(0, "../../src/")

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

    #
    # Some natural constants
    #

    au  = 1.49598e13     # Astronomical Unit       [cm]
    pc  = 3.08572e18     # Parsec                  [cm]
    ms  = 1.98892e33     # Solar mass              [g]
    ts  = 5.78e3         # Solar temperature       [K]
    ls  = 3.8525e33      # Solar luminosity        [erg/s]
    rs  = 6.96e10        # Solar radius            [cm]
    
    #
    # Grid parameters
    #

    nr       = 128
    ntheta   = 128
    nphi     = 1
    rin      = 1000 * rs
    rout     = 1000 * rin
    thetaup  = 0.

    #
    # Star parameters
    #

    mstar    = 4.76 * ms
    rstar    = 3.34 * rs
    tstar    = 16500.
    pstar    = np.array([0.,0.,0.])

    #
    # Disk parameters
    #

    agr = 1e-5  # grain radius [cm]
    rhogr = 3.71 # grain bulk density [g.cm-3]
    mgr = 4. * mt.pi * agr**3 * rhogr / 3. 
    ndin = 1.e-1 # numberdensity [cm-3]
    rhodin = ndin * mgr
    
    hin = 200 * rs
    powhd = 1.3
    powrhod = -2.22
    

    #
    # Make the coordinates
    #

    ri       = np.logspace(mt.log10(rin), mt.log10(rout), nr)
    thetai   = np.linspace(thetaup, 0.5*mt.pi, ntheta)

    #
    # Make the grid
    #

    qq       = np.meshgrid(ri, thetai, indexing='ij')
    rr       = qq[0] * np.sin(qq[1])
    tt       = qq[1]
    zr       = qq[0] * np.cos(qq[1])

    #
    # Make the dust density model
    #

    hd = hin * (rr/rin)**powhd
    rhod = np.empty((nr, ntheta))
    rhod[:, 1:] = rhodin * (rr[:, 1:]/rin)**powrhod * np.exp(-0.5 *(zr[:, 1:]/hd[:, 1:])**2)
    rhod[:, 0] = rhod[:, 1]

    #
    # Write the wavelength_micron.inp file
    #
    
    lammin = 1.e-8
    lammax = 1.e-2
    lam = np.logspace(mt.log10(lammin), mt.log10(lammax), 128)
    nlam     = lam.size

    #
    # Write the wavelength_micron.inp file
    #

    src = source.BlackBody(rstar, tstar, lam)

    #
    # Write the extinction opacity
    #

    opac = np.genfromtxt("run_ppdisk_simple_1/dustkappa_silicate.inp", skipheader=)

    
    
    # env = envelope.Envelope(lam, ri, thetai, Kext, S)

    # img.compute_intensity(env, src)
    # flux = img.compute_flux()

 
