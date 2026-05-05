#
# Import NumPy for array handling
#

import numpy as np
import math as mt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import sys
from astropy.io import ascii
import os  
sys.path.insert(0, "../../../src/")
import image
import planck
import constants as ct
import envelope
import source

from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.natconst import *

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
# Monte Carlo parameters
#

nphot    = 1e7

#
# Grid parameters
#

nr       = 128
ntheta   = 100
nphi     = 1
rin      = 1000 * rs
rout     = 10000 * rin
thetaup  = 0.

#
# Star parameters
#

mstar    = 4.76 * ms
rstar    = 3.34 * rs
tstar    = 16500.
pstar    = np.array([0.,0.,0.])


#
# Reading dust opacities
#

opac = np.genfromtxt("dustkappa_silicate.inp",skip_header=10)
Cabs = opac[:,1]
Csca = opac[:,2]
Cext = Cabs + Csca
Cext_max = np.max(Cext)

#
# Disk parameters
#

hin = 200 * rs
powhd = 1.3
powrhod = -2.22
tau_midplane = 100. # Vertical tau at the inner radius

gam = powrhod + 1.
rhodin = gam * tau_midplane / (Cext_max*rin*((rout/rin)**gam - 1.)) # density at the disc base and inner radius [g.cm-3]

#
# Make the coordinates
#

dr = 10. / (rhodin * Cext_max)
nr1 = 64
r1 = rin+dr

ri1 = np.linspace(rin, r1, nr1)
ri2 = np.logspace(mt.log10(r1), mt.log10(rout), nr-nr1+2)
ri = np.concatenate((ri1, ri2[1:]))

thetai   = np.arccos(np.linspace(1, 0, ntheta+1))
phii     = np.linspace(0., mt.pi*2., nphi+1)

rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )

#
# Make the grid
#

qq       = np.meshgrid(rc, thetac, phic, indexing='ij')
rr       = qq[0] * np.sin(qq[1])
tt       = qq[1]
zr       = qq[0] * np.cos(qq[1])

#
# Make the dust density model
#

hd = hin * (rr/rin)**powhd
rhod = rhodin * (rr/rin)**powrhod * np.exp(-0.5 *(zr/hd)**2)


#
# Write the wavelength_micron.inp file
#

lammin = 1.e-2 # minimal wavelenghth [microns]
lammax = 1.e4  # maximal wavelenghth [microns]
lam = np.logspace(mt.log10(lammin), mt.log10(lammax), 64)
nlam     = lam.size

#
# Write the wavelength file
#

with open('wavelength_micron.inp','w+') as f:
    f.write('%d\n'%(nlam))
    for value in lam:
        f.write('%13.6e\n'%(value))
        
#
# Write the stars.inp file
#

with open('stars.inp','w+') as f:
    f.write('2\n')
    f.write('1 %d\n\n'%(nlam))
    f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar, mstar, pstar[0], pstar[1], pstar[2]))
    for value in lam:
        f.write('%13.6e\n'%(value))
    f.write('\n%13.6e\n'%(-tstar))
    
#
# Write the grid file
#

with open('amr_grid.inp','w+') as f:
    f.write('1\n')                       # iformat
    f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
    f.write('100\n')                     # Coordinate system: spherical
    f.write('0\n')                       # gridinfo
    f.write('1 1 0\n')                   # Include r,theta coordinates
    f.write('%d %d %d\n'%(nr,ntheta,1))  # Size of grid
    for value in ri:
        f.write('%13.6e\n'%(value))      # X coordinates (cell walls)
    for value in thetai:
        f.write('%13.6e\n'%(value))      # Y coordinates (cell walls)
    for value in phii:
        f.write('%13.6e\n'%(value))      # Z coordinates (cell walls)
        
#
# Write the density file
#

with open('dust_density.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
    f.write('1\n')                       # Nr of dust species
    data = rhod.ravel(order='F')         # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')
    
#
# Dust opacity control file
#

with open('dustopac.inp','w+') as f:
    f.write('2               Format number of this file\n')
    f.write('1               Nr of dust species\n')
    f.write('============================================================================\n')
    f.write('1               Way in which this dust species is read\n')
    f.write('0               0=Thermal grain\n')
    f.write('silicate        Extension of name of dustkappa_***.inp file\n')
    f.write('----------------------------------------------------------------------------\n')
    
#
# Write the radmc3d.inp control file
#

with open('radmc3d.inp','w+') as f:
    f.write('nphot = %d\n'%(nphot))
    f.write('scattering_mode_max = 1\n')
    f.write('iranfreqmode = 1\n')
    f.write('istar_sphere = 1\n')

    
#os.system("radmc3d mcthem setthreads 12")

incl = 90.
PA = 0.

# os.system("radmc3d sed incl 60 phi 0")


#
# RTSPyCE
#

qq = np.meshgrid(ri, thetai, indexing='ij')
R = qq[0] * np.sin(qq[1])
z = qq[0] * np.cos(qq[1])


#
# Source class for RTSPYCE
#

src = source.BlackBody(rstar, tstar, lam*1e-6)

#
# Reading the radmc3d temperature
#

d = readData()
temp  = d.dusttemp[:,:,0,0]
interp = RegularGridInterpolator((d.grid.x, d.grid.y), temp, bounds_error=False, fill_value=None)

T = interp((qq[0], qq[1]))

B = planck.planck_function_freq(ct.c*1e6/lam, T)

#
# Make envelope class for RTSPYCE
#

hd = hin * (R/rin)**powhd
rhod = np.empty_like(R)
rhod[:, 1:] = rhodin * (R[:, 1:]/rin)**powrhod * np.exp(-0.5 *(z[:, 1:]/hd[:, 1:])**2)
rhod[:, 0] = rhod[:, 1]


f = interp1d(opac[:, 0], Cext, bounds_error=False, fill_value="extrapolate")
Cext_interp = f(lam)
f = interp1d(opac[:, 0], Cabs, bounds_error=False, fill_value="extrapolate")
Cabs_interp = f(lam)

Kext = rhod[None, :, :] * Cext_interp[:, None, None]

dtau = 0.5*(Kext[:, 1:] + Kext[:, :-1]) * (ri[None, 1:, None] - ri[None, :-1, None])
tau = np.empty_like(Kext)
tau[:, 0] = 0.
tau[:, 1:] = np.cumsum(dtau, axis=1)


J = 0.25 * (rstar/ri[None, :, None])**2 * src.intensity[:, None, None] * np.exp(-tau)

epsilon = Cabs_interp / Cext_interp
S = epsilon[:, None, None] * B + (1. - epsilon[:, None, None]) * J


zeros = np.zeros((nlam, 2, ntheta+1))
Kext = np.concatenate([zeros, Kext], axis=1)
S = np.concatenate([zeros, S], axis=1)
ri = np.insert(ri, 0, 0.99*rin)
ri = np.insert(ri, 0, rstar)


env = envelope.Envelope(lam*1e-6, ri, thetai, Kext, S)

d = ct.pc * 1e2

img = image.PolarImage(rstar, 1, rout, 128, 100, incl, PA, d, lam*1e-6)

img.compute_intensity(env, src)

flux = img.compute_flux()

ascii.write([flux], "flux_rtspyce.dat", overwrite=True, format="no_header")

