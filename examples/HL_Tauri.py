import sys
import numpy as np
import math as mt

sys.path.insert(0, "../src/")

from envelope import Envelope
import source
import image

# A few conversion factors ...

sun2au = 4.6524726e-3
au2sun = 1. / sun2au


class Disc(Envelope):

    def __init__(self, params):

        # Defining the circumstellar grid.
        # The grid must start at the stellar radius.
        # We treat the inner cavity with two points from Rstar to Rstar - epsilon, where S and Kext will be 0.
        
        r_cavity = np.linspace(params["rstar"], params["rin"] - 1e-3, 2)
        r_disc = np.logspace(mt.log10(params["rin"]), mt.log10(params["rout"]), params["nr"]-2)
        r = np.concatenate((r_cavity, r_disc))

        # Refining angular points around the disc mid-plane
        theta = np.acos(np.linspace(1., 0., params["ntheta"]))

        

        


if __name__ == "__main__":

    params = {

        "rstar": 1.,
        "rin": 0.1 * au2sun,
        "rout": 60 * au2sun,
        "nr": 8,
        "ntheta": 4
        
    }

    env = Disc(params)

    
