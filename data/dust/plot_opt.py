import glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("sil-draine.nk") + glob.glob("amC-hann.nk") + glob.glob("sil-dlee.nk")


fig, ax = plt.subplots(1, 2, figsize=(17, 8), layout="tight", sharex=True)

shift = 0.

for file in files:

    data = np.genfromtxt(file)

    ax[0].plot(data[:, 0], data[:, 1]+shift, marker="+", label=file)
    ax[1].plot(data[:, 0], data[:, 2]+shift, marker="+", label=file)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[0].legend()
plt.show()



    

