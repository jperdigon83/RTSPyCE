# * Free Software Foundation. 
# * 
# * This file is distributed in the hope that it will be   
# * useful, but WITHOUT ANY WARRANTY; without even the     
# * implied warranty of MERCHANTABILITY or FITNESS FOR A   
# * PARTICULAR PURPOSE. See the GNU General Public License 
# * for more details.
# * 
# * You should have received a copy of the GNU General     
# * Public License along with this program. If not, see    
# * <https://www.gnu.org/licenses/>.

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
import math as mt
from astropy.io import fits

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "../../src"))
sys.path.insert(0, parent_dir)

from rtspyce import RTSpyce
import planck
import constants as ct
from gas import freefree_boundfree_absorption


def comparison_disco(filename):

    hdul = fits.open(filename)[0]
    hdr = hdul.header
    
    # Stellar parameters
        
    Tstar = hdr['TSTAR']
    Rstar = hdr['RSTAR']
    Mstar = hdr['MSTAR']

    # Disc parameters
    
    Tin = hdr['TDISC0']
    rout = hdr['RDISC'] 
    ionisation_factor = hdr['IONFRAC']
    rho0 = hdr['DENS0']
    alpha = hdr['POWDENS']
    beta = hdr['POWH']
    gamma = hdr['POWTD']
  
    # Image parameters
    
    d = hdr['DISTANCE'] * ct.pc / ct.R_sun
    N = hdr['NAXIS1']
    L = 2*rout + d*hdr['CDELT1']
    incl = hdr['INCL']

    # Wavelength parameters
    
    nlam = hdr['NAXIS3']
    lam_min = float(hdr['CRVAL3'])
    lam_max = lam_min + float(hdr['CDELT3'])*(nlam-1)
    lam = np.linspace(lam_min, lam_max, nlam)
    nu = ct.c / lam

    # ---
    # Circumstellar grid
    # ---
    
    nr = 128
    ntheta = 64
    
    r = np.logspace(mt.log10(Rstar), mt.log10(rout), nr)
    theta = np.arccos(np.linspace(1., 0., ntheta))

    R = r[:, None] * np.sin(theta[None, :])
    Z = r[:, None] * np.cos(theta[None, :])

    # ---
    # Temperature
    # ---
    
    T = np.empty((nr, ntheta))
    T[:, 1:] = Tin * (R[:, 1:]/Rstar)**gamma
    T[:, 0] = T[:, 1]

    # ---
    # number density profile
    # ---

    mu = 1. / (1. + ionisation_factor)
    
    H0 = np.sqrt(ct.k_B * Rstar**3 * T * ct.R_sun / (mu * ct.m_p * ct.G * Mstar * ct.M_sun))

    H = H0 * (R/Rstar)**beta

    n0 = rho0 / (mu * ct.m_p)
   
    ndensity = np.empty_like(H)
    ndensity[:, 1:] = n0 * (R[:, 1:]/Rstar)**alpha * np.exp(-0.5*(Z[:, 1:]/H[:, 1:])**2)
    ndensity[:, 0] = 0.

    # ---
    # Extinction coefficient
    # ---

    ne = ionisation_factor * ndensity
    Kext = freefree_boundfree_absorption(nu, ne, T) * ct.R_sun

    # ---
    # Stellar intensity
    # ---

    Istar = planck.planck_function_freq(nu, Tstar)

    # ---
    # Circumstellar source function
    # ---
    
    S = planck.planck_function_freq(nu, T)
  
    # ---
    # Cartesian image
    # ---
   
    dx = L/N
    x = np.linspace(0.5*(dx-L), 0.5*(L-dx), N)
    y = np.copy(x)
    x, y = np.meshgrid(x, y, indexing='ij')
    x = np.ravel(x)
    y = np.ravel(y)

    foo = RTSpyce(r, theta)
    
    intensity = foo.intensity_map(x, y, incl, S, Kext, Istar)
    intensity = intensity.reshape(nlam, N, N)

    intensity_disco = np.transpose(hdul.data, [0, 2, 1]) * lam[:, None, None]**2/ct.c
    
    diff = 100*(intensity - intensity_disco) / intensity_disco
    
    for m in range(nlam):
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
        fontsize = 20
        
        extent = [-0.5*L/Rstar, 0.5*L/Rstar, -0.5*L/Rstar, 0.5*L/Rstar]
        norm = cm.Normalize(np.min([intensity_disco[m], intensity[m]]), np.max([intensity_disco[m], intensity[m]]))
        
        plt.suptitle(r' $\lambda =$ ' + str(np.round(lam[m]*1e6, 2)) + r' $\mu m$', fontsize=fontsize)
        
        ax[0].set_title('DISCO')
        ax[0].imshow(intensity_disco[m].T, origin='lower', cmap='afmhot', extent=extent, norm=norm)
       
        ax[1].set_title('RTSpyce')
        ax[1].imshow(intensity[m].T, origin='lower', cmap='afmhot', extent=extent, norm=norm)
       
        ax[2].set_title('Relative differences (%)')
        c = ax[2].imshow(np.transpose((diff[m])), origin='lower', cmap='afmhot', extent=extent)
        plt.colorbar(c, ax=ax[2], fraction=0.046, pad=0.04)

        fig.supxlabel(r"$x \, \mathrm{[R_\star]}$", fontsize=fontsize)
        fig.supylabel(r"$y \, \mathrm{[R_\star]}$", fontsize=fontsize)
 
if __name__ == "__main__":

    comparison_disco("M02_20-100microns_incl=0deg.fits")
    
    plt.show()
        
      
        
