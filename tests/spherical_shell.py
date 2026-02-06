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

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math as mt

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, parent_dir)

from rtspyce import RTSpyce
import planck
import constants as ct

class SphericalShell:

    def __init__(self, r_in, r_out):

        self.r_in = r_in
        self.r_out = r_out
        self.r2_in = r_in**2
        self.r2_out = r_out**2
        
    def intensity(self, x, y, tau, I_star, S_env):

        assert len(x) == len(y)
        assert len(tau) == len(I_star)

        p2 = x*x + y*y
        nnu = len(tau)
        nx = len(x)

        intensity = np.zeros((nnu, nx))

        mask_star = p2 < 1.
        mask_cavity = (p2 >= 1.) & (p2 < self.r2_in)
        mask_env = (p2 >= self.r2_in) & (p2 < self.r2_out)

        Kin = tau / (self.r_out - self.r_in)

        tau_star = Kin[:, None] * (np.sqrt(self.r2_out - p2[mask_star]) - np.sqrt(self.r2_in - p2[mask_star]))
        tau_cavity = Kin[:, None] * (np.sqrt(self.r2_out - p2[mask_cavity]) - np.sqrt(self.r2_in - p2[mask_cavity]))
        tau_env = Kin[:, None] * np.sqrt(self.r2_out - p2[mask_env])

        intensity[:, mask_star] = S_env[:, None] + np.exp(-tau_star)*(I_star[:, None] - S_env[:, None])
        intensity[:, mask_cavity] = S_env[:, None] * (1. - np.exp(-2.*tau_cavity))
        intensity[:, mask_env] = S_env[:, None] * (1. - np.exp(-2.*tau_env))

        return intensity
    
def comparison_rtspyce_sphericalshell(incl):
    
    r_in = 2.
    r_out = 10.
    T_star = 2500.
    T_env = 800.
    n = 128
    nnu = 7

    x = np.linspace(-r_out, r_out, n)
    y = np.copy(x)

    x, y = np.meshgrid(x, y, indexing="ij")
    x = np.ravel(x)
    y = np.ravel(y)
    
    tau = np.logspace(-3, 3, nnu)
    nu = np.logspace(1, 7, nnu) * ct.c
    I_star = planck.planck_function_freq(nu, T_star)

    
    S_env = planck.planck_function_freq(nu, T_env)

    foo = SphericalShell(r_in, r_out)
    intensity_shell = foo.intensity(x, y, tau, I_star, S_env)
    intensity_shell = intensity_shell.reshape(nnu, n, n)

    # ---
    # The intensity from RTSpyce
    # ---

    nr = 128
    ntheta = 64

    r_cavity = np.linspace(1., r_in-1e-5, 2)
    r_env = np.logspace(mt.log10(r_in), mt.log10(r_out), nr)
    r = np.concatenate((r_cavity, r_env))
    
    theta = np.linspace(0., 0.5*mt.pi, ntheta)

    T_cavity = np.zeros((2, ntheta))
    T_env = T_env * np.ones((nr, ntheta))
    T = np.concatenate((T_cavity, T_env), axis=0)
    S = planck.planck_function_freq(nu, T)

    Kext_cavity = np.zeros((nnu, 2, ntheta))
    Kin = tau / (r_out - r_in)
    Kext_env = Kin[:, None, None] * np.ones((nnu, nr, ntheta))
    Kext = np.concatenate((Kext_cavity, Kext_env), axis=1)
    
    foo1 = RTSpyce(r, theta)
    intensity_rtspyce = foo1.intensity_map(x, y, incl, S, Kext, I_star)
    intensity_rtspyce = intensity_rtspyce.reshape(nnu, n, n)

    fig, ax = plt.subplots(3, nnu, figsize=(14, 6), layout="constrained")

    c1 = []
    c2 = []
    c3 = []

    diff = np.abs(intensity_rtspyce - intensity_shell)/intensity_shell
    diff[diff != diff] = 0.
    for i in range(nnu):

        img1 = intensity_shell[i].T
        img2 = intensity_rtspyce[i].T
        img3 = diff[i].T
        
        c1.append(ax[0, i].imshow(img1, origin="lower", norm=LogNorm(1e-100, np.max(intensity_shell), clip=True)))
        c2.append(ax[1, i].imshow(img2, origin="lower", norm=LogNorm(1e-100, np.max(intensity_rtspyce), clip=True)))
        c3.append(ax[2, i].imshow(img3, origin="lower", norm=LogNorm(1e-16, np.max(diff), clip=True)))

    fig.colorbar(c1[0], ax=ax[0, :], orientation='vertical')
    fig.colorbar(c2[0], ax=ax[1, :], orientation='vertical')
    fig.colorbar(c3[0], ax=ax[2, :], orientation='vertical')
    
if __name__ == "__main__":

    """ Test with an analytical case: a spherical shell with constant temperature and constant optical extinction. """
    
    # comparison_rtspyce_sphericalshell(0.)
    # comparison_rtspyce_sphericalshell(45.)
    comparison_rtspyce_sphericalshell(80.)

    plt.show()
    
