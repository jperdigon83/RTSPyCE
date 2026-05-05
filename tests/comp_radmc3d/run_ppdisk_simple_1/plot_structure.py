import numpy as np
import math as mt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.natconst import *

au  = 1.49598e13     # Astronomical Unit       [cm]
rs  = 6.96e10        # Solar radius            [cm]

#
# Make sure to have done the following beforhand:
#
#  First compile RADMC-3D
#  Then run:
#   python problem_setup.py
#   radmc3d mctherm
#

#
# Read the data
#

d     = readData()
rr, tt = np.meshgrid(d.grid.x,d.grid.y,indexing='ij')
R = rr * np.sin(tt)
z   = rr * np.cos(tt)
rhod  = d.rhodust[:,:,0,0]
temp  = d.dusttemp[:,:,0,0]


#
# View a surface plot of the density structure
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# ax.plot_surface(np.log10(rr)/au, zzr, np.log10(rhod), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)

#
# Plot the vertical density structure at different radii
#

# irr = [0,10,20,-1]
# plt.figure()
# for ir in irr:
#     r    = d.grid.x[ir]
#     rstr = '{0:4.0f}'.format(r/au)
#     rstr = 'r = '+rstr.strip()+' au'
#     plt.semilogy(tt[ir,:]/mt.pi, rhod[ir,:], label=rstr)
# plt.ylim((1e-25,1e-15))
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$\rho_{\mathrm{dust}}\;[\mathrm{g}/\mathrm{cm}^3]$')
# plt.legend()

#
# Plot the azimutal temperature at different radii
#


plt.figure()
for ir in range(0, d.grid.nx, 5):
    r    = d.grid.x[ir]
    rstr = '{0:.3f}'.format(r/rs)
    rstr = 'r = '+rstr.strip()+' au'
    plt.plot(tt[ir,:]/mt.pi, temp[ir,:], label=rstr)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$T_{\mathrm{dust}}\;[\mathrm{K}]$')
plt.legend()

#
# Plot the radial temperature structure at different azimut
#


plt.figure()
for ir in range(0, d.grid.ny, 5):
    t    = d.grid.y[ir]
    rstr = '{0:4.0f}'.format(np.rad2deg(t))
    rstr = 't = '+rstr.strip()+' deg'
    plt.plot(rr[:, ir]/au,temp[:, ir])
plt.xlabel(r'$r ~ \mathrm{[au]}$')
plt.ylabel(r'$T_{\mathrm{dust}}\;[\mathrm{K}]$')
plt.loglog()

#
# Plot the temperature map
#

plt.figure()
rin = d.grid.x[0]
hin = 200 * rs
c = plt.pcolormesh(R/rs, z/rs, temp, cmap="afmhot", zorder=1)
levels = np.linspace(np.min(temp), np.max(temp), 16)
plt.contour(R/rs, z/rs, temp, levels=levels, colors="white",zorder=2)
plt.colorbar(c)
plt.xlim(rin/rs, 4*rin/rs)
plt.xlabel(r"$R ~ [R_\mathrm{\star}]$")
plt.xlabel(r"$z ~ [R_\mathrm{\star}]$")
plt.ylim(0, 3*rin/rs)



#
# Plot the density map
#

plt.figure()
rin = d.grid.x[0]
hin = 200 * rs
c = plt.pcolormesh(R/rs, z/rs, rhod, cmap="afmhot", zorder=1)
levels = np.linspace(np.min(rhod), np.max(rhod), 16)
plt.contour(R/rs, z/rs, rhod, levels=levels, colors="white",zorder=2)
plt.colorbar(c)
plt.xlim(rin/rs, 4*rin/rs)
plt.xlabel(r"$R ~ [R_\mathrm{\star}]$")
plt.xlabel(r"$z ~ [R_\mathrm{\star}]$")
plt.ylim(0, 3*rin/rs)


plt.show()


