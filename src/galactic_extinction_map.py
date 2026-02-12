import numpy as np
import math as mt
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import trapezoid

class DustMap:
    
    def __init__(self, dustmap_filename, extinction_filename):

        hdul = fits.open(dustmap_filename)

        mean_ext = hdul[1].data
        hdr = hdul[1].header
        std_ext = hdul[2].data

        self.mean_ext = np.moveaxis(mean_ext, 0, -1)
        self.err_ext = np.moveaxis(std_ext, 0, -1)
    
        # Constructing longitude vector

        lonMin = hdr['CRVAL1'] - (hdr['CRPIX1'] - 1)*hdr['CDELT1']
        lonMax = hdr['CRVAL1'] + (hdr['NAXIS1'] - (hdr['CRPIX1']-1))*hdr['CDELT1']
        nlon = hdr['NAXIS1']

        self.lon_map = np.linspace(lonMin, lonMax, nlon)
    
        # Construction latitude vector

        latMin = hdr['CRVAL2'] - (hdr['CRPIX2']-1)*hdr['CDELT2']
        latMax = hdr['CRVAL2'] + (hdr['NAXIS2'] - (hdr['CRPIX2']-1))*hdr['CDELT2']
        nlat = hdr['NAXIS2']

        self.lat_map = np.linspace(latMin, latMax, nlat)
    
        # Constructing distance vector

        dMin = hdr['CRVAL3']
        dMax = hdr['CRVAL3'] + (hdr['NAXIS3']-1)*hdr['CDELT3']
        nd = hdr['NAXIS3']

        self.dist_map = np.linspace(dMin, dMax, nd)
        
        # Loading the extinction curve
      
        self.ext_curve = np.genfromtxt(extinction_filename)
        
        self.ext_curve[:, 0] *= 1.0e-9


    def compute_extinction_curve(self, lon_star, lat_star, dist_star):

        if lon_star > 180.:
            lon_star -= 360.

        # Interpolating the differential extiction at the stellar coordinates
    
        interp = RegularGridInterpolator((self.lat_map, self.lon_map), self.mean_ext)
        mean_ext_star = interp((lat_star, lon_star))

        interp = RegularGridInterpolator((self.lat_map, self.lon_map), self.err_ext)
        std_ext_star = interp((lat_star, lon_star))
    
        # Integrating the differential extinction to the stellar distance (nasty integral)
    
        idx = (d <= dist_star) & (mean_ext_star == mean_ext_star)

        mean_ext_star = trapezoid(mean_ext_star[idx], self.dist_map[idx])
        std_ext_star = trapezoid(std_ext_star[idx], self.dist_map[idx])
        
        A = mean_ext_star * self.ext_curve[:, 1]
        std_A = std_ext_star * self.ext_curve[:, 1]

        return A, std_A


    def derreddening_spectrum(self, A, std_A, wave, spectrum, sigma_spectrum):
        
        interp = interp1d(self.ext_curve[:, 0], np.log(A), bounds_error=False, fill_value=(np.log(A[0]), np.log(A[-1])))
   
        A = np.exp(interp(wave))
   
        interp = interp1d(self.ext_curve[:, 0], std_A, bounds_error=False, fill_value=(std_A[0], std_A[-1]))
    
        std_A = interp(wave)

        spectrum_corr = spectrum * 10**(0.4*A)
        
        sigma_spectrum_corr = 10**(0.4*A) * (sigma_spectrum + 0.4*mt.log(10.) * spectrum * std_A)

        return spectrum_corr, sigma_spectrum_corr

    
    def reddening_spectrum(self, A, std_A, wave, spectrum, sigma_spectrum):
        
        interp = interp1d(self.ext_curve[:, 0], np.log(A), bounds_error=False, fill_value=(np.log(A[0]), np.log(A[-1])))
   
        A = np.exp(interp(wave))
   
        interp = interp1d(self.ext_curve[:, 0], std_A, bounds_error=False, fill_value=(std_A[0], std_A[-1]))
    
        std_A = interp(wave)

        spectrum_corr = spectrum * 10**(-0.4*A)
        
        sigma_spectrum_corr = 10**(-0.4*A) * (sigma_spectrum + 0.4*mt.log(10.) * spectrum * std_A)

        return spectrum_corr, sigma_spectrum_corr

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    data = np.genfromtxt("/home/jperdigon/projects/massif/HD50138/data/flux/HD50138_flux.dat")
    
    foo = DustMap("mean_and_std_lbd.fits", "extinction_curve.txt")

    l = 219.1517633045930
    b = -3.1441503310724
    d = 351
    
    A, std_A = foo.compute_extinction_curve(l, b, d)
    spec, sigma = foo.derreddening_spectrum(A, std_A, data[:, 0], data[:, 1], data[:, 2])

    spec_1, sigma_1 = foo.reddening_spectrum(A, std_A, data[:, 0], data[:, 1], data[:, 2])

    plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2])
    plt.errorbar(data[:, 0], spec, yerr=sigma, label="derreddened")
    plt.errorbar(data[:, 0], spec_1, yerr=sigma_1, label="reddened")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()

    

    
    
        
