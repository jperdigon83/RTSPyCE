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
import sys

rt_dir = '/home/jperdigon/projects/rt_spyce'
sys.path.append(rt_dir + '/src')

from rtspyce import RTSpyce
import planck
import stellar_scattering
import constants as ct
import stellar_radiation
import gas

def power_law_integration(x, y):

    assert np.all(np.diff(x) > 0.)

    y[y < np.finfo(float).tiny] = np.finfo(float).tiny
    
    x_ratio = x[1:] / x[:-1]
    
    y[y < np.finfo(float).tiny] = np.finfo(float).tiny

    y_ratio = y[..., 1:]/y[..., :-1]
    
    beta = np.log(y_ratio) / np.log(x_ratio) + 1.

    assert np.all(beta != 0.)

    dI = y[..., :-1] * x[:-1] * (x_ratio**beta - 1.) / beta

    return np.sum(dI, axis=-1)

class GasAndDustDisc:

    def __init__(self, params):

        # Loading wavelenghts and frequencies
     
        self.wavelengths = wavelengths
        self.nu = ct.c / wavelengths

        # The wavelength at which we constrain the optical depth
        
        wavelengths_0 = 1.0e-6
        nu_0 = ct.c / wavelengths_0

        # ---
        #
        # PARAMETERS
        #
        # ---
      
        self.R_star = params["R_star"]
        self.Teff_star = params["Teff_star"]
        self.log_g_star = params["log(g_star)"]
        self.mh_star = params["[M/H]_star"]
        self.dist = params["distance"] * ct.pc / ct.R_sun

        self.tau_gas = params["tau_gas"]
        self.pow_n_gas = params["pow_n_gas"]
        self.pow_H_gas = params["pow_H_gas"]
        self.Hin_gas = params["Hin_gas"]
        self.Tin_gas = params["Tin_gas"]
        self.pow_T_gas = params["pow_T_gas"]

        self.Rin_dust = params["Rin_dust"]
        self.Rout_dust = params["Rout_dust"]
        self.tau_dust = params["tau_dust"]
        self.pow_n_dust = params["pow_n_dust"]
        self.pow_H_dust = params["pow_H_dust"]
        self.Hin_dust = params["Hin_dust"]
        self.Tin_dust = params["Tin_dust"]
        self.pow_T_dust = params["pow_T_dust"]
        self.log10_grain_radius_min = params["log10(amin)"]
        self.log10_grain_radius_max = params["log10(amax)"]
        self.pow_grain_radii = params["pow_a"]

        self.incl = params["incl"]
        self.PA = np.deg2rad(params["PA"])

        # Grid parameters

        n_theta = 128
        nr_dust = 64
        nr_gas = 32

        self.theta = np.arccos(np.linspace(1., 0., n_theta))

        cos_theta, sin_theta = np.cos(self.theta), np.sin(self.theta)
        
        # ---
        #
        # Gas disc
        #
        # ---

        Rin_Rstar = self.Rin_dust / self.R_star
        
        r_gas = np.logspace(mt.log10(self.R_star), mt.log10(self.Rin_dust - 1e-5), nr_gas)
        
        R_gas = r_gas[:, None] * sin_theta[None, :]
        z_gas = r_gas[:, None] * cos_theta[None, :]

        T_gas = np.empty_like(R_gas)
        T_gas[:, 1:] = self.Tin_gas * (R_gas[:, 1:]/self.R_star)**self.pow_T_gas
        T_gas[:, 0] = T_gas[:, 1]

        H_gas = self.Hin_gas * (R_gas/self.R_star)**self.pow_H_gas

        integrand = (1. - np.exp(-ct.h * nu_0 / (ct.k_B * T_gas[:, -1]))) * (r_gas/self.R_star)**(2. * self.pow_n_gas) * gas.gaunt_factor(np.array([wavelengths_0]), T_gas[:, -1])[0] / np.sqrt(T_gas[:, -1])
        
        A = 3.692349106159646e-2 * ct.R_sun * power_law_integration(r_gas, integrand) / nu_0**3

        if self.pow_n_gas != -1.:

            k = self.pow_n_gas + 1.
            B = ct.sigma_T * self.R_star * ct.R_sun * ((Rin_Rstar)**k - 1.) / k
            
        else:
            
            B = ct.sigma_T * self.R_star * ct.R_sun * mt.log(Rin_Rstar)

        C = - self.tau_gas
        
        nin_gas = 0.5 * (- B + mt.sqrt(B*B - 4.*A*C)) / A

        ne_gas = np.empty_like(R_gas)
        ne_gas[:, 1:] = nin_gas * (R_gas[:, 1:]/self.R_star)**self.pow_n_gas * np.exp(-0.5*(z_gas[:, 1:]/H_gas[:, 1:])**2)
        ne_gas[:, 0] = ne_gas[:, 1]

        Kabs_gas = gas.hydrogen_freefree_boundfree_absorption(self.nu, ne_gas, T_gas) * ct.R_sun
        Ksca_gas = gas.thompson_scattering(ne_gas) * ct.R_sun
        Kext_gas = Kabs_gas + Ksca_gas

        epsilon_gas = np.empty_like(Kext_gas)
        idx = Kext_gas > np.finfo(float).tiny
        epsilon_gas[idx] = Kabs_gas[idx] / Kext_gas[idx]
        epsilon_gas[~idx] = 0.

        # ---
        #
        # Dust disc
        #
        # ---
        
        r_dust = np.logspace(mt.log10(self.Rin_dust), mt.log10(self.Rout_dust), nr_dust)
        
        R_dust = r_dust[:, None] * sin_theta[None, :]
        z_dust = r_dust[:, None] * cos_theta[None, :]

        T_dust = np.empty_like(R_dust)
        T_dust[:, 1:] = self.Tin_dust * (R_dust[:, 1:]/self.Rin_dust)**self.pow_T_dust
        T_dust[:, 0] = T_dust[:, 1]

    
        grain_radii = np.logspace(self.log10_grain_radius_min, self.log10_grain_radius_max, 128)

        grain_radii_dist = (grain_radii/grain_radii[0])**self.pow_grain_radii

        Cabs, Csca = dust.dust_opacities(refidx_astroSil, wavelengths, grain_radii, wgt=grain_radii_dist, extrapolate=True)
        Cabs0, Csca0 = dust.dust_opacities(refidx_astroSil, np.array([wavelengths_0]), grain_radii, wgt=grain_radii_dist, extrapolate=True)
        Cext = Cabs + Csca
        Cext0 = Cabs0 + Csca0
        
        # Number density

        H_dust = np.empty_like(R_dust)
        H_dust[:, 1:] = self.Hin_dust * (R_dust[:, 1:]/self.Rin_dust)**self.pow_H_dust
        H_dust[:, 0] = H_dust[:, 1]

        Rout_Rin = self.Rout_dust / self.Rin_dust
        if self.pow_n_dust != -1.:

            k = self.pow_n_dust + 1.
            
            nin_dust = k * self.tau_dust / (Cext0 * self.Rin_dust * ((Rout_Rin)**k - 1.))
            
        else:
            
            nin_dust = self.tau_dust / (Cext0 * self.Rin_dust * mt.log(Rout_Rin))

        n_dust = np.empty_like(R_dust)
        n_dust[:, 1:] = nin_dust * (R_dust[:, 1:]/self.Rin_dust)**self.pow_n_dust * np.exp(-0.5*(z_dust[:, 1:]/H_dust[:, 1:])**2)
        n_dust[:, 0] = n_dust[:, 1]

        Kext_dust = n_dust[None, :, :] * Cext[:, None, None]
        epsilon_dust = np.ones_like(Kext_dust) * Cabs[:, None, None] / Cext[:, None, None]


        # ---
        #
        # STELLAR INCIDENT RADIATION
        #
        # ---
        
        self.I_star = stellar_radiation.stellar_radiation(wavelengths, "ck04models", self.Teff_star, self.mh_star, self.log_g_star) * wavelengths**2 / ct.c
      

        # ---
        #
        # Assembling components
        #
        # ---

        self.r = np.concatenate((r_gas, r_dust))

        self.Kext = np.concatenate((Kext_gas, Kext_dust), axis=1)
        
        epsilon = np.concatenate((epsilon_gas, epsilon_dust), axis=1)

        T = np.concatenate((T_gas, T_dust), axis=0)

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
        nnu = len(self.wavelengths)

        foo = RTSpyce(self.r, self.theta)
        
        intensityMap = foo.intensity_map(np.ravel(self.x[:, :N_v_h]), np.ravel(self.y[:, :N_v_h]), self.incl, self.S, self.Kext, self.I_star)
        
        self.image = np.empty((nnu, self.N_u, self.N_v))
        
        self.image[:, :, :N_v_h] = intensityMap.reshape(nnu, self.N_u, N_v_h)
        self.image[:, :, N_v_h:] = self.image[:, :, :N_v_h][:, :, ::-1]
        
        # Converting x, y, and dS in radians, for the computation of physical observables

        self.image = np.reshape(self.image, (nnu, self.N_u*self.N_v))
        
        self.x = np.ravel(self.x) / self.dist
        
        self.y = np.ravel(self.y) / self.dist
        
        self.dS = np.ravel(0.5 * (self.uw[1:, None]**2 - self.uw[:-1, None]**2) * (self.vw[None, 1:] - self.vw[None, :-1])) / self.dist**2

        # Computing the observed flux
        
        self.flux = np.sum(self.image * self.dS[None, :], axis=-1)



        
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
        nnu = len(self.wavelengths)

        foo = RTSpyce(self.r, self.theta)
        
        intensityMap = foo.intensity_map(np.ravel(self.x[:Nh, :]), np.ravel(self.y[:Nh, :]), self.incl, self.S, self.Kext, self.I_star)
      
        self.image = np.empty((nnu, N, N))
        
        self.image[:, :Nh, :] = intensityMap.reshape(nnu, Nh, N)
        
        self.image[:, Nh:, :] = self.image[:, :Nh, :][:, ::-1, :]

        # Flattening the arrays
    
        # Converting x, y, and dS in radians, for the computation of physical observables
        
        dS = (dx / self.dist)**2
        self.dS = np.full_like(self.x, dS)
        self.x /= self.d
        self.y /= self.d

        # Computing the observed flux

        self.flux = np.sum(self.image, axis=(1, 2)) * dS



        
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from observables import Observables

    obs_L = Observables()
    wave_L = np.array([3.4e-6, 3.6e-6, 3.8e-6])
    obs_L.load_data("../data/L-band/Chop/old_reduction/", np.min(wave_L), np.max(wave_L))

    obs_N = Observables()
    wave_N = np.linspace(9e-6, 11e-6, 8)
    obs_N.load_data('../data/N-band/old_reduction/', np.min(wave_N), np.max(wave_N))

    sed_data = np.genfromtxt("../data/flux/flux_unred.dat", comments='"')

    params_model = {"R_star": 4.,
                    "Teff_star": 21800.,
                    "log(g_star)": 4.0,
                    "[M/H]_star": 0.,
                    "distance": 572.,

                    "tau_gas": 1e2,
                    "pow_n_gas": -3.5,
                    "pow_H_gas": 1.5,
                    "Hin_gas": 0.1,
                    "Tin_gas": 20000,
                    "pow_T_gas": -0.75,
                
                    "Rin_dust": 1000.,
                    "Rout_dust": 1e9,
                    "tau_dust": 1e1,
                    "pow_n_dust": -2.0,
                    "pow_H_dust": 1.3,
                    "Hin_dust": 150.,
                    "Tin_dust": 1200.,
                    "pow_T_dust": -0.70,
                    "log10(amin)": -8.,
                    "log10(amax)": -5.,
                    "pow_a": -3.5,
                    "incl": 53.,
                    "PA": 20.}

    flux_wavelengths = np.logspace(mt.log10(sed_data[0, 0]), mt.log10(sed_data[-1, 0]), 64)
    wavelengths = np.concatenate((flux_wavelengths, wave_L, wave_N))
    wavelengths = np.unique(wavelengths)
    wavelengths = np.sort(wavelengths)
    
     
    # ----
    # Computing physical observables
    # ---
    
    model = GasAndDustDisc(params_model, wavelengths)
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

    nu_data = ct.c / sed_data[:, 0] 
    ax.errorbar(sed_data[:, 0], nu_data*sed_data[:, 1], yerr=nu_data*sed_data[:, 2], fmt=".", alpha=0.3, zorder=1, color="k")
    ax.plot(model.wavelengths, model.flux*model.nu, zorder=2, color="r")

    ax.set_xscale("log")
    ax.set_yscale("log")


    

    
    plt.show()
    
