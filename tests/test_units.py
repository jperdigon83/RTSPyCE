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

import matplotlib.pyplot as plt
import numpy as np
import math as mt
import sys
import os
import time

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, "../src"))
sys.path.insert(0, parent_dir)

from rtspyce import RTSpyce
import constants as ct
import planck

def test_intersections(r, theta, x, y, incl):

    foo = RTSpyce(r, theta)
    n = len(x)

    t = np.empty(100)
    
    for i in range(100):
        
        t0 = time.time()
        s, r_s, theta_s, idx_star = foo.intersections_with_grid(x, y, incl)
        t1 = time.time()
        t[i] = t1 - t0

    print("incl:", incl, "Mean(time)", np.mean(t), "std(time)", np.std(t))
    
    fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
    
    for i in range(n):
       
        ax[0].plot(s[i], r_s[i])
        ax[1].plot(s[i], theta_s[i]/mt.pi)

def test_interpolation(r, theta, S, Kext, x, y, incl):

    n = len(x)
    
    foo = RTSpyce(r, theta)

    s, r_s, theta_s, idx_star = foo.intersections_with_grid(x, y, incl)
     
    t = np.empty(100)
    
    for i in range(100):

        t0 = time.time()
        S_s, Kext_s = foo.interpolation_along_rays(r_s, theta_s, S, Kext)
        t1 = time.time()
        t[i] = t1 - t0

    print("incl:", incl, "Mean(time)", np.mean(t), "std(time)", np.std(t))

    fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
    
    for i in range(n):
      
        ax[0].plot(s[i], S_s[i][0])
        ax[1].plot(s[i], Kext_s[i][0])

    ax[0].set_yscale("log")
    #ax[1].set_yscale("log")


def test_tau_along_rays(r, theta, S, Kext, x, y, incl):

    n = len(x)
    foo = RTSpyce(r, theta)

    s, r_s, theta_s, idx_star = foo.intersections_with_grid(x, y, incl)

    S_s, Kext_s = foo.interpolation_along_rays(r_s, theta_s, S, Kext)
     
    t = np.empty(100)
    
    for i in range(100):

        t0 = time.time()
        dtau_s, tau_s = foo.tau_along_rays(s, Kext_s)
        t1 = time.time()
        t[i] = t1 - t0

    print("incl:", incl, "Mean(time)", np.mean(t), "std(time)", np.std(t))

    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(10, 10))

    for i in range(n):
      
        ax.plot(s[i], tau_s[i][0])
        ax.plot(s[i], tau_s[i][1])


def test_intensity_along_rays(nu, r, theta, S, Kext, x, y, incl):

    n = len(x)
    foo = RTSpyce(r, theta)

    s, r_s, theta_s, idx_star = foo.intersections_with_grid(x, y, incl)

    S_s, Kext_s = foo.interpolation_along_rays(r_s, theta_s, S, Kext)

    dtau_s, tau_s = foo.tau_along_rays(s, Kext_s)

    I_star = planck.planck_function_freq(nu, 5800.)
    intensity = foo.integration_along_rays(dtau_s, tau_s, S_s, Kext_s, idx_star, I_star)
       
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(10, 10))

    for i in range(n):
      
        ax.plot(nu, intensity[:, i])
    ax.set_yscale("log")

if __name__ == "__main__":

    log_Rin = 0.0
    log_Rout = 2.0
    Rin = mt.exp(log_Rin)
    Rout = mt.exp(log_Rout)
    
    r = np.logspace(log_Rin, log_Rout, 200)
    theta = np.linspace(0.0, 0.5*mt.pi, 100)
    nu = np.logspace(3., 8., 2) * ct.c

    nr = len(r)
    ntheta = len(theta)
    nnu = len(nu)

    x = np.array([-10., 0., 1., 5.])
    y = np.copy(x)
    
    T = 1000. * np.ones((nr, ntheta))
    S = planck.planck_function_freq(nu, T)

    tau = 0.01
    Kext = tau/(Rout - Rin) * np.ones((nnu, nr, ntheta))

    incl = np.array([0., 20., 45., 80., 90.])
    
    for i in incl:
        
        # test_intersections(i)
        # test_interpolation(r, theta, S, Kext, x, y, i)
        # test_tau_along_rays(r, theta, S, Kext, x, y, i)
        test_intensity_along_rays(nu, r, theta, S, Kext, x, y, i)

        
    plt.show()

    

    

    
