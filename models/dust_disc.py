# * ----       This file is part of the project        ----
# * 
# * Copyright (C) 2024 Perdigon, J.. All Rights Reserved.
# * 
# * This file is licensed under the terms of the GNU       
# * General Public License, version 3., as published by the
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

import numpy as np
import math as mt
from scipy.interpolate import interp1d
import sys
import os 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(parent_dir)

from rtspyce import RTSpyce
import planck
import stellar_scattering
import constants as ct
import stellar_radiation

class DustDisc:

    def __init__(self, params):

        # ---
        #
        # PARAMETERS
        #
        # ---

        self.wavelengths = params["wavelengths"]
        self.wavelengths_0 = params["reference_wavelength"]
      
        self.R_star = params["R_star"]
        self.Teff_star = params["Teff_star"]
        self.log_g_star = params["log(g_star)"]
        self.mh_star = params["[M/H]_star"]
        self.d = params["distance"] * ct.pc / ct.R_sun
        
        self.Rin_dust = params["Rin_dust"]
        self.Rout_dust = params["Rout_dust"]
        self.tau_dust = params["tau_dust"]
        self.pow_n_dust = params["pow_n_dust"]
        self.pow_H_dust = params["pow_H_dust"]
        self.Hin_dust = params["Hin_dust"]
        self.Tin_dust = params["Tin_dust"]
        self.pow_T_dust = params["pow_T_dust"]
        self.Cabs_dust = params["Cabs_dust"]
        self.Csca_dust = params["Csca_dust"]
       
        self.incl = params["incl"]
        self.PA = np.deg2rad(params["PA"])

        # Frequencies vector

        self.nu = ct.c / self.wavelengths
        self.nnu = len(self.nu)

        
        # Grid parameters

        n_theta = 128
        nr_dust = 64

        self.theta = np.arccos(np.linspace(1., 0., n_theta))

        cos_theta, sin_theta = np.cos(self.theta), np.sin(self.theta)
        
        # ---
        #
        # Cavity
        #
        # ---

        r_cavity = np.linspace(self.R_star, self.Rin_dust - 1e-5, 2)

        T_cavity = np.zeros((2, n_theta))

        Kext_cavity = np.zeros((self.nnu, 2, n_theta))

        epsilon_cavity = np.ones_like(Kext_cavity)
        
        # ---
        #
        # Dust disc
        #
        # ---

        # Radial grid

        Rout_Rin = self.Rout_dust/self.Rin_dust
        log_Rout_Rin = mt.log(Rout_Rin)
        
        r_dust = self.Rin_dust * np.exp(log_Rout_Rin * np.linspace(0., 1., nr_dust))
        R_dust = r_dust[:, None] * sin_theta[None, :]
        z_dust = r_dust[:, None] * cos_theta[None, :]

        # Temperature

        T_dust = np.empty_like(R_dust)
        T_dust[:, 1:] = self.Tin_dust * (R_dust[:, 1:]/self.Rin_dust)**self.pow_T_dust
        T_dust[:, 0] = T_dust[:, 1]

        
        Cabs0 = np.exp(interp1d(np.log(self.wavelengths), np.log(self.Cabs_dust))(mt.log(self.wavelengths_0)))
        Csca0 = np.exp(interp1d(np.log(self.wavelengths), np.log(self.Csca_dust))(mt.log(self.wavelengths_0)))
        Cext0 = Cabs0 + Csca0
        
        # Number density

        H_dust = np.empty_like(R_dust)
        H_dust[:, 1:] = self.Hin_dust * (R_dust[:, 1:]/self.Rin_dust)**self.pow_H_dust
        H_dust[:, 0] = H_dust[:, 1]
        
        if self.pow_n_dust != -1.:

            k = self.pow_n_dust + 1.
            
            nin_dust = k * self.tau_dust / (Cext0 * self.Rin_dust * ((Rout_Rin)**k - 1.))
            
        else:
            
            nin_dust = self.tau_dust / (Cext0 * self.Rin_dust * log_Rout_Rin)

        n_dust = np.empty_like(R_dust)
        n_dust[:, 1:] = nin_dust * (R_dust[:, 1:]/self.Rin_dust)**self.pow_n_dust * np.exp(-0.5*(z_dust[:, 1:]/H_dust[:, 1:])**2)
        n_dust[:, 0] = n_dust[:, 1]

        Cext = self.Cabs_dust + self.Csca_dust
        
        Kext_dust = n_dust[None, :, :] * Cext[:, None, None]

        epsilon_dust = np.ones_like(Kext_dust) * self.Cabs_dust[:, None, None] / Cext[:, None, None]

        # ---
        #
        # STELLAR INCIDENT RADIATION
        #
        # ---
        
        self.I_star = stellar_radiation.stellar_radiation(self.wavelengths, "ck04models", self.Teff_star, self.mh_star, self.log_g_star) * self.wavelengths**2 / ct.c
      

        # ---
        #
        # Assembling components
        #
        # ---

        self.r = np.concatenate((r_cavity, r_dust))

        self.Kext = np.concatenate((Kext_cavity, Kext_dust), axis=1)
        
        epsilon = np.concatenate((epsilon_cavity, epsilon_dust), axis=1)

        T = np.concatenate((T_cavity, T_dust), axis=0)

        B = planck.planck_function_freq(self.nu, T)
        
        J_star = stellar_scattering.compute_stellar_mean_intensity(self.R_star, self.I_star, self.r, self.Kext)

        self.S = epsilon * B + (1. - epsilon) * J_star

        
    def polar_images(self, params_test=(2, 51, 51, 51)):

        Rin_Rstar = self.Rin_dust / self.R_star
        log_Rin_Rstar = mt.log(Rin_Rstar)
        
        Rout_Rin = self.Rout_dust / self.Rin_dust
        log_Rout_Rin = mt.log(Rout_Rin)
        
        # Building the radial coordinate u of the image
        
        uw_star = np.linspace(0., self.R_star, params_test[0])
        u_star = 0.5*(uw_star[1:] + uw_star[:-1])
        uw_star = uw_star[:-1]

        uw_gas = np.linspace(0., 1., params_test[1])
        u_gas = 0.5*(uw_gas[1:] + uw_gas[:-1])
        uw_gas = self.R_star * np.exp(log_Rin_Rstar * uw_gas)
        u_gas = self.R_star * np.exp(log_Rin_Rstar * u_gas)
        uw_gas = uw_gas[:-1]
        
        uw_disc = np.linspace(0., 1., params_test[2])
        u_disc = 0.5*(uw_disc[1:] + uw_disc[:-1])
        uw_disc = self.Rin_dust * np.exp(log_Rout_Rin * uw_disc)
        u_disc = self.Rin_dust * np.exp(log_Rout_Rin * u_disc)

        self.uw = np.concatenate((uw_star, uw_gas, uw_disc))
        self.u = np.concatenate((u_star, u_gas, u_disc))
        self.N_u = len(self.u)

        # Building the angular coordinate v of the image
        
        self.vw = np.linspace(0, 2*mt.pi, params_test[3])
        self.v = 0.5*(self.vw[1:] + self.vw[:-1])
        self.N_v = len(self.v)
        
        self.u, self.v = np.meshgrid(self.u, self.v, indexing='ij')

        self.x = self.u * np.sin(self.v)
        self.y = self.u * np.cos(self.v)

        # Computing the half the image with the ray-tracing routine, for efficiency purpose
        N_v_h = int(0.5*self.N_v)

        foo = RTSpyce(self.r, self.theta)
        
        intensityMap = foo.intensity_map(np.ravel(self.x[:, :N_v_h]), np.ravel(self.y[:, :N_v_h]), self.incl, self.S, self.Kext, self.I_star)
        
        self.image = np.empty((self.nnu, self.N_u, self.N_v))
        
        self.image[:, :, :N_v_h] = intensityMap.reshape(self.nnu, self.N_u, N_v_h)
        self.image[:, :, N_v_h:] = self.image[:, :, :N_v_h][:, :, ::-1]
        
        # Converting x, y, and dS in radians, for the computation of physical observables

        self.image = np.reshape(self.image, (self.nnu, self.N_u*self.N_v))
        
        self.x = np.ravel(self.x) / self.d
        
        self.y = np.ravel(self.y) / self.d
        
        self.dS = np.ravel(0.5 * (self.uw[1:, None]**2 - self.uw[:-1, None]**2) * (self.vw[None, 1:] - self.vw[None, :-1])) / self.d**2

        # Computing the observed flux
        
        self.flux = np.sum(self.image*self.dS[None, :], axis=-1)

    def cartesian_images(self, N, width):

        dx = width / N
        
        # Computing x and y coordinates of the pixels
        
        self.xw = np.linspace(-0.5*width, 0.5*width, int(N+1))
        self.yw = np.linspace(-0.5*width, 0.5*width, int(N+1))

        self.x = 0.5 * (self.xw[1:] + self.xw[:-1])
        self.y = 0.5 * (self.yw[1:] + self.yw[:-1])

        self.x, self.y = np.meshgrid(self.x, self.y, indexing='ij')
        self.xw, self.yw = np.meshgrid(self.xw, self.yw, indexing='ij')
       
        # Computing the half of the image with the ray-tracing routine, for efficiency purpose

        Nh = int(0.5*N)

        foo = RTSpyce(self.r, self.theta)
        
        intensityMap = foo.intensity_map(np.ravel(self.x[:Nh, :]), np.ravel(self.y[:Nh, :]), self.incl, self.S, self.Kext, self.I_star)
      
        self.image = np.empty((self.nnu, N, N))
        
        self.image[:, :Nh, :] = intensityMap.reshape(self.nnu, Nh, N)
        
        self.image[:, Nh:, :] = self.image[:, :Nh, :][:, ::-1, :]

        # Flattening the arrays
    
        # Converting x, y, and dS in radians, for the computation of physical observables
        
        dS = (dx / self.d)**2
        self.dS = np.full_like(self.x, dS)
        self.x /= self.d
        self.y /= self.d

        # Computing the observed flux

        self.flux = np.sum(self.image, axis=(1, 2)) * dS


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from observables import Observables
    from dust import opacities
    
    obs_L = Observables()
    wave_L = np.array([3.4e-6, 3.6e-6, 3.8e-6])
    obs_L.load_data("/home/jperdigon/projects/massif/l_Pup/data/MATISSE_LM/", np.min(wave_L), np.max(wave_L))

    obs_N = Observables()
    wave_N = np.linspace(9e-6, 11e-6, 8)
    obs_N.load_data('/home/jperdigon/projects/massif/l_Pup/data/MATISSE_N/', np.min(wave_N), np.max(wave_N))

    sed_data = np.genfromtxt("/home/jperdigon/projects/massif/l_Pup/data/HD62623_sed.dat", comments='"')

    wavelengths = np.concatenate((sed_data[:, 0], wave_L, wave_N))
    wavelengths = np.unique(wavelengths)
    wavelengths = np.sort(wavelengths)

    refractive_index_list = [np.genfromtxt("/home/jperdigon/projects/rt_spyce/data/dust/amC_Hanner.nk"), np.genfromtxt("/home/jperdigon/projects/rt_spyce/data/dust/crMgFeSil.nk")]
    
    grain_radii = np.logspace(-8, -3, 128)
    grain_radii_list = [grain_radii, grain_radii]
    
    grain_dist = (grain_radii/grain_radii[0])**(-3.5)
  
    grain_dist_list = [grain_dist, grain_dist]

    grain_prop_list = np.array([0.5, 0.5])
    
    foo = opacities(wavelengths, refractive_index_list, grain_radii_list, grain_dist_list, grain_prop_list)

    plt.plot(wavelengths, foo[0], label="Cabs")
    plt.plot(wavelengths, foo[1], label="Csca")
    plt.xscale("log")
    plt.yscale("log")
    
    params_model = {"wavelengths": wavelengths,
                    "reference_wavelength": 1e-6,
                    
                    "R_star": 54.,
                    "Teff_star": 8500.,
                    "log(g_star)": 2.,
                    "[M/H]_star": 0.,
                    "distance": 630.,
                  
                    "Rin_dust": 685.,
                    "Rout_dust": 10**(7.7),
                    "tau_dust": 10**1,
                    "pow_n_dust": -1.55,
                    "pow_H_dust": 1.26,
                    "Hin_dust": 114.,
                    "Tin_dust": 1200.,
                    "pow_T_dust": -0.82,
                    "Cabs_dust": foo[0],
                    "Csca_dust": foo[1],
                    "incl": 41.5,
                    "PA": 269.}

    # ----
    # Computing physical observables
    # ---
    
    model = DustDisc(params_model)
    
    model.polar_images()
    
    obs_L.compute_visibilities(model, True, 1.8)
    obs_L.compute_closure_phases()
    
    obs_N.compute_visibilities(model, True, 1.8)
    obs_N.compute_closure_phases()

    # ---
    # Ploting visibilities and closure phases
    # ---

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), layout="constrained")
    elw = 0.3
    
    for i in range(obs_L.n_files):
        
        for j in range(6):

            idx = obs_L.vis2_data[i][j] == obs_L.vis2_data[i][j]
            x = obs_L.baseline_data[i][j] / obs_L.wavelengths_data[i]

            ax[0].plot(x[idx], obs_L.vis2_model[i][j][idx], color="b", zorder=3)
            
            ax[0].errorbar(x[idx], obs_L.vis2_data[i][j][idx], yerr=obs_L.err_vis2_data[i][j][idx], fmt='.', color='k', elinewidth=elw, zorder=2, markersize=0.5)

        for j in range(4):

            idx = obs_L.t3phi_data[i][j] == obs_L.t3phi_data[i][j]
            x = obs_L.triplet_perimeter_data[i][j] / obs_L.wavelengths_data[i]
            
            ax[1].plot(x[idx], obs_L.t3phi_model[i][j][idx], color="b", zorder=3)
    
            ax[1].errorbar(x[idx], obs_L.t3phi_data[i][j][idx], yerr=obs_L.err_t3phi_data[i][j][idx], fmt='.', color='k', elinewidth=elw, zorder=2, markersize=0.5)

    for i in range(obs_N.n_files):
        
        for j in range(6):

            idx = obs_N.vis2_data[i][j] == obs_N.vis2_data[i][j]
            x = obs_N.baseline_data[i][j] / obs_N.wavelengths_data[i]

            ax[0].plot(x[idx], obs_N.vis2_model[i][j][idx], color="r", zorder=3)
            
            ax[0].errorbar(x[idx], obs_N.vis2_data[i][j][idx], yerr=obs_N.err_vis2_data[i][j][idx], fmt='.', color='k', elinewidth=elw, zorder=2, markersize=0.5)

        for j in range(4):

            idx = obs_N.t3phi_data[i][j] == obs_N.t3phi_data[i][j]
            x = obs_N.triplet_perimeter_data[i][j] / obs_N.wavelengths_data[i]
            
            ax[1].plot(x[idx], obs_N.t3phi_model[i][j][idx], color="r", zorder=3)
    
            ax[1].errorbar(x[idx], obs_N.t3phi_data[i][j][idx], yerr=obs_N.err_t3phi_data[i][j][idx], fmt='.', color='k', elinewidth=elw, zorder=2, markersize=0.5)
            

    ax[0].set_yscale("log")
    
    # ---
    # Plotting SED
    # ---
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

    ax.errorbar(sed_data[:, 0], sed_data[:, 1], yerr=sed_data[:, 2], fmt=".")
    ax.plot(model.wavelengths, model.flux*model.nu)

    ax.set_xscale("log")
    ax.set_yscale("log")



    plt.show()
    
