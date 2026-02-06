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
import constants as ct
import hydrogen

a0 = 5.291772109e-11
E0 = 13.6*1.602177e-19
E0kB = E0 / ct.k_B
n03 = 0.5*0.55396 / a0
K0 = ct.h**2 / (2*mt.pi*ct.m_e*ct.k_B)

def hydrogen_partition_function(n, T):

    if T <= 1e-306:
        
        return 2.

    else:

        k_max = int(2**24)
        U = 1.

        alpha = - 4. * mt.pi * n * a0*a0*a0 / 3.
        beta = - E0 / (ct.k_B * T)
        
        for k in range(2, k_max+1):
            
            k2 = k * k
          
            dU = k2 * mt.exp(alpha * (1 + k2)**3) * mt.exp(beta * (1. - 1./k2))

            if dU/U < 1e-5:
              
                break
    
            elif k == k_max:
                
                print("Warning, you have reached the maximal number of iterations !")
                
                U += dU

            else:
                
                U += dU
            
        U *= 2.

        return U
        
#%%

def ionisation_equilibrium(n, T):

    if np.shape(n) != np.shape(T):
        raise RuntimeError('np.shape(n) != np.shape(T)')

    if len(np.shape(n)) != 0:

        idx = n > np.finfo(float).tiny
        x = np.empty_like(n)

        x[~idx] = np.where(T[~idx] > 222.5, 1., 0.)

        func = np.vectorize(hydrogen_partition_function)
        u = func(n[idx], T[idx])

        with np.errstate(over='ignore'):
            
            F = 0.5 * n[idx] * u * np.exp(E0kB/T[idx]) * (K0/T[idx])**1.5
        
        x[idx] = 2./(1. + np.sqrt(1. + 4.*F))

        return x
  
    else:

        if n > np.finfo(float).tiny:

            u = hydrogen_partition_function(n, T)

            with np.errstate(over='ignore'):
                
                F = 0.5 * n * u * np.exp(E0kB/T) * (K0/T)**1.5

            return 2./(1. + np.sqrt(1. + 4.*F))

        else:

            return 1.
           
    

#%%

if __name__=="__main__":

    import matplotlib.pyplot as plt
    
    T = np.logspace(0, 8, 128)
    n = 1e12
 
    import time

    t0 = time.time()

    U_Fortran = np.array([hydrogen.hydrogen_partition_function(n, t) for t in T])

    t1 = time.time()

    t_Fortran = t1 - t0
    print("Fortran : ", t_Fortran)

    t0 = time.time()
    
    U_Python = np.array([hydrogen_partition_function(n, t) for t in T])

    t1 = time.time()

    t_Python = t1 - t0
    
    print("Python : ", t_Python)

    print(t_Python / t_Fortran)

    plt.plot(T, U_Fortran, label="Fortran")
    plt.plot(T, U_Python, label="Python")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()
