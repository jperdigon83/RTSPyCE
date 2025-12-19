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
 
import math as mt
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

class RTSPyCE:

    def __init__(self, r, theta):

        assert (theta[0] == 0.) and (theta[-1] == 0.5*mt.pi)
        
        self.r = r
        self.theta = theta
        self.r2 = r*r
        
        self.nr = len(r)
        self.ntheta = len(theta)

    def intensity_map(self, x, y, incl, S, Kext, I_star):
        
        """ A method to compute directly the intensity map from the distribution of pixels
        """
        s, r_s, theta_s, idx_star = self.intersections_with_grid(x, y, incl)
        
        S_s, Kext_s = self.interpolation_along_rays(r_s, theta_s, S, Kext)

        dtau_s, tau_s = self.tau_along_rays(s, Kext_s)

        return self.integration_along_rays(dtau_s, tau_s, S_s, Kext_s, idx_star, I_star)

    def intersections_with_grid(self, x, y, incl):

        incl = np.deg2rad(incl)
        n = len(x)
        x2 = x*x
        y2 = y*y
        
        s = [[] for _ in range(n)]
        r_s = [[] for _ in range(n)]
        theta_s = [[] for _ in range(n)]

        cos_incl = mt.cos(incl)
        sin_incl = mt.sin(incl)
        tan_incl = mt.tan(incl)
        cos2_incl = cos_incl**2
        sin2_incl = sin_incl**2
        
        mask_env = (x2 + y2) < self.r2[-1]  # mask for pixels that intersect the envelope
        idx_env = np.nonzero(mask_env)[0]

        mask_star = (x2 + y2) < self.r2[0]  # mask for pixels that intersect the star
        idx_star = np.nonzero(mask_star)[0]
       
        # ---
        # Intersections with radii
        # ---
        
        for idx in idx_env:
            for i in range(self.nr-1, -1, -1):

                delta = self.r2[i] - x2[idx] - y2[idx]
                
                if delta > 0:

                    s_value = mt.sqrt(delta)
                    theta_value = self.theta_coordinate(cos_incl, sin_incl, -s_value, y[idx], self.r[i])

                    s[idx].append(-s_value)
                    r_s[idx].append(self.r[i])
                    theta_s[idx].append(theta_value)

                    if not mask_star[idx]:

                        theta_value = self.theta_coordinate(cos_incl, sin_incl, s_value, y[idx], self.r[i])
                        
                        s[idx].append(s_value)
                        r_s[idx].append(self.r[i])
                        theta_s[idx].append(theta_value)
                        
                else:
                    
                    break

        # ---
        # Intersections with cones
        # ---
        
        tan2_theta = np.tan(self.theta)**2
        B = sin2_incl - cos2_incl * tan2_theta
        C = cos_incl * sin_incl * (tan2_theta + 1.)
        
        for idx in idx_env:
            for i in range(1, self.ntheta-1):
                
                delta = tan2_theta[i] * y2[idx] - B[i] * x2[idx]

                if delta > 0.:

                    s1 = (-C[i]*y[idx] - mt.sqrt(delta)) / B[i]
                    s2 = (-C[i]*y[idx] + mt.sqrt(delta)) / B[i]
                    
                    if not mask_star[idx]:

                        r2_s_1 = self.r2_coordinate(x2[idx], y2[idx], s1**2)
                        
                        if (r2_s_1 > self.r2[0]) and (r2_s_1 < self.r2[-1]):

                            s[idx].append(s1)
                            r_s[idx].append(mt.sqrt(r2_s_1))
                            theta_s[idx].append(self.theta[i])

                        r2_s_2 = self.r2_coordinate(x2[idx], y2[idx], s2**2)
                        
                        if (r2_s_2 > self.r2[0]) and (r2_s_2 < self.r2[-1]):

                            s[idx].append(s2)
                            r_s[idx].append(mt.sqrt(r2_s_2))
                            theta_s[idx].append(self.theta[i])

                    else:
                        
                        s_val = min(s1, s2)

                        if s_val < 0:
                    
                            r2_s = self.r2_coordinate(x2[idx], y2[idx], s_val**2)

                            if (r2_s > self.r2[0]) and (r2_s < self.r2[-1]):

                                s[idx].append(s_val)
                                r_s[idx].append(mt.sqrt(r2_s))
                                theta_s[idx].append(self.theta[i])


        # ---
        # Intersections with the equatorial plane
        # ---
        
        for idx in idx_env:

            s_val = tan_incl*y[idx]

            if not mask_star[idx]:

                r2_s = self.r2_coordinate(x2[idx], y2[idx], s_val**2)
                
                if (r2_s > self.r2[0]) and (r2_s < self.r2[-1]):

                    s[idx].append(s_val)
                    r_s[idx].append(mt.sqrt(r2_s))
                    theta_s[idx].append(0.5*mt.pi)

            else:

                if (s_val < 0):
                    
                    r2_s = self.r2_coordinate(x2[idx], y2[idx], s_val**2)

                    if (r2_s > self.r2[0]) and (r2_s < self.r2[-1]):
                        
                        s[idx].append(s_val)
                        r_s[idx].append(mt.sqrt(r2_s))
                        theta_s[idx].append(0.5*mt.pi)

        # ---
        # Sorting the arrays along s
        # ---
        
        s = [np.array(lst) for lst in s]
        r_s = [np.array(lst) for lst in r_s]
        theta_s = [np.array(lst) for lst in theta_s]

        idx_sort = [np.argsort(x) for x in s]
        
        s = [s[i][idx_sort[i]] for i in range(n)]
        r_s = [r_s[i][idx_sort[i]] for i in range(n)]
        theta_s = [theta_s[i][idx_sort[i]] for i in range(n)]

        return s, r_s, theta_s, idx_star

    def interpolation_along_rays(self, r_s, theta_s, S, Kext):

        n = len(r_s)
    
        swap_S = np.moveaxis(S, 0, 2)
        swap_Kext = np.moveaxis(Kext, 0, 2)

        log_r = np.log(self.r)
        
        func_S = RGI((log_r, self.theta), swap_S)
        func_Kext = RGI((log_r, self.theta), swap_Kext)

        S_s = [func_S((np.log(r_s[idx]), theta_s[idx])) for idx in range(n)]
        Kext_s = [func_Kext((np.log(r_s[idx]), theta_s[idx])) for idx in range(n)]
        
        S_s = [np.moveaxis(S_s[idx], 1, 0) for idx in range(n)]
        Kext_s = [np.moveaxis(Kext_s[idx], 1, 0) for idx in range(n)]

        return S_s, Kext_s

    def tau_along_rays(self, s, Kext_s):

        n = len(Kext_s)

        Kext_mid = [0.5 * (Kext_s[idx][:, 1:] + Kext_s[idx][:, :-1]) for idx in range(n)]
        ds = [s[idx][1:] - s[idx][:-1] for idx in range(n)]
        
        dtau_s = [ds[idx][None, :] * Kext_mid[idx] for idx in range(n)]
       
        tau_s = [np.cumsum(dtau_s[idx], axis=-1) for idx in range(n)]

        tau_s = [np.insert(tau_s[idx], 0, 0., axis=-1) if np.all(np.array(Kext_s[idx].shape) != 0) else Kext_s[idx] for idx in range(n)]

        return dtau_s, tau_s
    
    def integration_along_rays(self, dtau_s, tau_s, S_s, Kext_s, idx_star, I_star):

        n = len(Kext_s)
        nnu = Kext_s[0].shape[0]
        
        exp_dtau_s = [np.exp(-dtau_s[idx]) for idx in range(n)]
        exp_tau_s = [np.exp(-tau_s[idx]) for idx in range(n)]

        intensity = np.empty((nnu, n))

        for i in range(n):

            idx = dtau_s[i] > 1e-5

            delta_intensity = np.empty_like(dtau_s[i])

            delta_intensity[idx] = exp_tau_s[i][:, :-1][idx] * (S_s[i][:, 1:][idx]*(1. - (1. + dtau_s[i][idx])*exp_dtau_s[i][idx]) + S_s[i][:, :-1][idx] * (dtau_s[i][idx] - 1. + exp_dtau_s[i][idx]))/dtau_s[i][idx]

            delta_intensity[~idx] = 0.5 * exp_tau_s[i][:, :-1][~idx] * dtau_s[i][~idx]*(S_s[i][:, :-1][~idx] + S_s[i][:, 1:][~idx])
            
            intensity[:, i] = np.sum(delta_intensity, axis=-1)

        # ---
        # Adding the itensity of the star in the image
        # ---
        
        for idx in idx_star:

            intensity[:, idx] += I_star * exp_tau_s[idx][:, -1]

        return intensity

    def r2_coordinate(self, x2, y2, s2):

        return s2 + x2 + y2

    def theta_coordinate(self, cos_incl, sin_incl, s, y, r):
        
        z = sin_incl*y - cos_incl*s

        zr = z/r

        return mt.asin(mt.sqrt(1. - zr*zr))
