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
import glob
from astropy.io import fits
from scipy.interpolate import interp1d
from image import Image

class Observables:
    
    def __init__(self, dataDir, lamMin, lamMax, verbose=False):

        assert lamMin < lamMax

        self.lamMin = lamMin
        self.lamMax = lamMax
         
        self.n_files = 0
        self.wavelengths_data = []
        self.u_data = []
        self.v_data = []
        self.baseline_data = []
        self.vis2_data = []
        self.err_vis2_data = []
        self.t3phi_data = []
        self.err_t3phi_data = []
        
        self.sta_baseline_data = []
        self.sta_triplet_data = []
        self.triplet_perimeter_data = []

        self.n_baselines = []
        self.n_triplets = []

        fileList = glob.glob(dataDir + '/*.fits')
        
        for file in fileList:

            if verbose is True:

                print("Processing file: " + file)
                
            hdul = fits.open(file)
            
            wavelengths = hdul['OI_WAVELENGTH'].data['EFF_WAVE']
            
            idx = (wavelengths >= self.lamMin) & (wavelengths <= self.lamMax)
            
            if np.any(idx):

                self.n_files += 1
                
                lambda_data = wavelengths[idx]
                ucoord = hdul['OI_VIS2'].data['UCOORD']
                vcoord = hdul['OI_VIS2'].data['VCOORD']
                
                self.wavelengths_data.append(lambda_data)
                self.u_data.append(ucoord)
                self.v_data.append(vcoord)

                baseline = np.sqrt(ucoord**2 + vcoord**2)
                
                self.baseline_data.append(baseline)
                self.n_baselines.append(len(baseline))
                
                # Loading visibilities

                vis2 = hdul['OI_VIS2'].data['VIS2DATA'][:, idx]
                err_vis2 = hdul['OI_VIS2'].data['VIS2ERR'][:, idx]
                vis2_flag_idx = hdul['OI_VIS2'].data['FLAG'][:, idx] is True

                vis2[vis2_flag_idx] = np.nan
                err_vis2[vis2_flag_idx] = np.nan
                
                self.vis2_data.append(vis2)
                self.err_vis2_data.append(err_vis2)
                
                # Loading closure phases

                t3phi = hdul['OI_T3'].data['T3PHI'][:, idx]
                err_t3phi = hdul['OI_T3'].data['T3PHIERR'][:, idx]
                t3phi_flag_idx = hdul['OI_T3'].data['FLAG'][:, idx] is True

                t3phi[t3phi_flag_idx] = np.nan
                err_t3phi[t3phi_flag_idx] = np.nan
                
                self.t3phi_data.append(t3phi)
                self.err_t3phi_data.append(err_t3phi)

                # Loading STA indexes for telescopes baselines and triangles

                sta_baseline = hdul['OI_VIS2'].data['STA_INDEX']
                self.sta_baseline_data.append(sta_baseline)

                sta_triplet = hdul['OI_T3'].data['STA_INDEX']
                self.sta_triplet_data.append(sta_triplet)

                n_triplets = len(sta_triplet)
                self.n_triplets.append(n_triplets)

                triplet_perimeter = np.empty(n_triplets)

                for j in range(n_triplets):

                    idx1 = (sta_baseline[:, 0] == sta_triplet[j, 0]) & (sta_baseline[:, 1] == sta_triplet[j, 1])
                   
                    idx2 = (sta_baseline[:, 0] == sta_triplet[j, 1]) & (sta_baseline[:, 1] == sta_triplet[j, 2])

                    idx3 = (sta_baseline[:, 0] == sta_triplet[j, 0]) & (sta_baseline[:, 1] == sta_triplet[j, 2])

                    triplet_perimeter[j] = baseline[idx1][0] + baseline[idx2][0] + baseline[idx3][0]

                self.triplet_perimeter_data.append(triplet_perimeter)

                
            else:
                
                print('Warning: lambda array for ' + file + ' not in [lamMin, lamMax], the file will we ignored.')


    def compute_visibilities(self, img: Image, apodisation=False, telescopeDiameter=None):

        self.vis_model = []
        self.vis2_model = []
        
        # ---
        # SELECTING MODELS IN THE WAVELENGTHS BAND OF OBSERVATIONS
        # ---
      
        idx = (img.wave >= self.lamMin) & (img.wave <= self.lamMax)
      
        assert np.count_nonzero(idx) > 1

        wavelengths_model = img.wave[idx]
        images_model = img.intensity[idx]

        # ---
        # APODIZATION WITH AN ESTIMATE OF THE PSF OF THE TELESCOPE
        # ---

        if apodisation is True:
            
            sigma_psf = wavelengths_model / (2.355*telescopeDiameter)
            images_model *= np.exp(-0.5*(img.x[None, :]**2 + img.y[None, :]**2)/ (img.d**2 * sigma_psf[:, None]**2))
            
        # ---
        # COMPUTATION OF THE OBSERVED FLUX
        # ---

        flux_model = np.sum(img.intensity[idx, :]*img.dpix, axis=-1) / img.d**2
    
        idx = flux_model > np.finfo(float).tiny

        # ---
        # COMPUTATION OF THE VISIBILITIES
        # ---
        
        PA = np.deg2rad(img.PA)
        
        cosPA, sinPA = mt.cos(PA), mt.sin(PA)

        for i in range(self.n_files):
            
            u_model = cosPA * self.u_data[i] - sinPA * self.v_data[i]
            v_model = sinPA * self.u_data[i] + cosPA * self.v_data[i]
            
            alpha = img.x[None, None, :] / img.d
            beta = img.y[None, None, :] / img.d
            
            phase = -2 * mt.pi * 1j * (alpha*u_model[:, None, None] + beta*v_model[:, None, None]) / wavelengths_model[None, :, None]

            V_model = np.zeros((self.n_baselines[i], len(wavelengths_model)), dtype=complex)
          
            V_model[:, idx] = np.sum(images_model[None, idx, :] * np.exp(phase[:, idx]) * img.dpix[None, None, :], axis=-1) / (img.d**2 * flux_model[None, idx])

            # Linearly interpolating complex visibilities of the model on the data wavelengths
            
            Vreal_interp = interp1d(np.log(wavelengths_model), V_model.real, axis=-1)
            Vimag_interp = interp1d(np.log(wavelengths_model), V_model.imag, axis=-1)
            
            Vreal = Vreal_interp(np.log(self.wavelengths_data[i]))
            Vimag = Vimag_interp(np.log(self.wavelengths_data[i]))

            Vreal2, Vimag2 = Vreal*Vreal, Vimag*Vimag
            
            V = Vreal + 1j * Vimag

            self.vis_model.append(V)
            self.vis2_model.append(Vreal2 + Vimag2)

    def compute_closure_phases(self):

        self.t3phi_model = []
        
        for i in range(self.n_files):

            t3phi = np.empty((self.n_triplets[i], len(self.wavelengths_data[i])))
            
            for j in range(self.n_triplets[i]):

                idx1 = (self.sta_baseline_data[i][:, 0] == self.sta_triplet_data[i][j, 0]) & (self.sta_baseline_data[i][:, 1] == self.sta_triplet_data[i][j, 1])
                   
                idx2 = (self.sta_baseline_data[i][:, 0] == self.sta_triplet_data[i][j, 1]) & (self.sta_baseline_data[i][:, 1] == self.sta_triplet_data[i][j, 2])

                idx3 = (self.sta_baseline_data[i][:, 0] == self.sta_triplet_data[i][j, 0]) & (self.sta_baseline_data[i][:, 1] == self.sta_triplet_data[i][j, 2])

                vis_product = np.ravel(self.vis_model[i][idx1] * self.vis_model[i][idx2] * np.conjugate(self.vis_model[i][idx3]))

                idx = (vis_product.real**2 + vis_product.imag**2) == vis_product.real**2
                
                t3phi[j, ~idx] = np.angle(vis_product[~idx], deg=True)
                
                t3phi[j, idx] = np.angle(vis_product.real[idx], deg=True)
   
            self.t3phi_model.append(t3phi)
            

    def compute_chi2_vis2(self):

        self.chi2_vis2 = 0.
        self.nchi2_vis2 = 0
        
        for i in range(self.n_files):

            diff2 = ((self.vis2_model[i] - self.vis2_data[i]) / self.err_vis2_data[i])**2
            diff2 = diff2[diff2 == diff2]
            
            self.nchi2_vis2 += len(diff2)
            self.chi2_vis2 += np.sum(diff2)
        
        if self.chi2_vis2 <= 0:

            print("Warning chi2 vis2 <= 0")
            self.chi2_vis2 = -np.inf
            self.nchi2_vis2 = 0
            
    def compute_chi2_t3(self):

        self.chi2_t3 = 0.
        self.nchi2_t3 = 0
        
        for i in range(self.n_files):
                
            diff = np.abs(self.t3phi_model[i] - self.t3phi_data[i])
                
            idx = diff > 180.
            diff[idx] = 360. - diff[idx]
                
            diff2 = (diff / self.err_t3phi_data[i])**2
            diff2 = diff2[diff2 == diff2]
                
            self.nchi2_t3 += len(diff2)
            self.chi2_t3 += np.sum(diff2)

        if self.chi2_t3 <= 0.:

            print("Warning chi2 t3 <= 0")
            self.chi2_t3 = -np.inf
            self.nchi2_t3 = 0
