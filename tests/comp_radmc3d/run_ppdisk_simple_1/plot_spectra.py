import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.natconst import *

flux = np.genfromtxt("flux_rtspyce.dat")
print(flux.shape)

fig3  = plt.figure()
s     = readSpectrum()
lam   = s[:,0]
nu    = 1e4*cc/lam
fnu   = s[:,1]
nufnu = nu*fnu
plt.plot(lam,nufnu*1e-3, label="radmc3d")
plt.plot(lam,nu*flux, label="rtspyce")
plt.xscale('log')
plt.yscale('log')
#plt.axis([1e-1, 1e4, 1e-10, 1e-4])
plt.xlabel('$\lambda\; [\mu \mathrm{m}$]')
plt.ylabel('$\\nu F_\\nu W m-2$')
plt.legend()



plt.show()

