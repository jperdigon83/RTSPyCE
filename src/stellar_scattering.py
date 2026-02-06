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

def compute_stellar_mean_intensity(Rstar, Istar, r, Kext):
    """
    Computes the mean intensity of the stellar light, at each point of the
    envelope grid.
    
        Parameters:
                   Rstar (double): Radius of the star
                   Istar (1D-array): The stellar incident radiation upon
                   the envelope

        Returns:
                mean_stellar_intensity (3D-array): the mean itensity of
                the stellar light
    """
    
    muStar = np.sqrt(1. - (Rstar/r)**2)

    dtau = 0.5*(Kext[:, 1:] + Kext[:, :-1]) * (r[None, 1:, None] - r[None, :-1, None])

    tau = np.empty_like(Kext)
    tau[:, 0] = 0.
    tau[:, 1:] = np.cumsum(dtau, axis=1)

    return 0.5 * (1 - muStar[None, :, None]) * Istar[:, None, None] * np.exp(-tau)
