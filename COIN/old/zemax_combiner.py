import os
import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt

# Set input values
delta_x = 0.000001
offset = 11
pixel_shift = 35
num_positions = 365
cycles = 3
flux_ratio = 1
dark_noise = 20

# Set simulation values
zmx_pix = 1.625e-6
cam_pix = 6.5e-6
ratio = zmx_pix/cam_pix

# Get files
path = os.getcwd()
file_names = np.array(os.listdir(path + "/Zemax_files/npy/"))

# Load PSFS
PSFs_large = np.empty([21, 1024, 1024])
for i in range(len(file_names)):
    idx = int(file_names[i].split("_")[0])
    PSFs_large[idx] = np.load(path + "/Zemax_files/npy/{}".format(file_names[i]))

# Set positions
Ts = np.linspace(0, 2*np.pi, num=num_positions, endpoint=False)
Xs = delta_x*np.sin(Ts)

# Get on axis PSF and scale by flux ratio
PSF_on_axis = ndimage.zoom(PSFs_large[0], ratio)
PSF_on_axis = flux_ratio * PSF_on_axis/np.max(PSF_on_axis)

# Generate shifted PSFs
PSFs_out = []

for i in range(1):
    # Shift
    PSF_off = ndimage.shift(PSFs_large[offset], [0, (pixel_shift + Xs[i])/ratio])
    # NOTE Y SHIFTS WORK IN THE OPPSOSITE DIRECTIONS

    # Interpolate and normalise
    PSF_off = ndimage.zoom(PSF_off, ratio)
    PSF_off = PSF_off/np.max(PSF_off)

    # Combine and normalise
    PSF = PSF_on_axis + PSF_off
    PSF = PSF/np.max(PSF)

    # Output
    PSFs_out.append(PSF)
    
PSF = PSF-0.005
PSF[PSF<=0] = 0

plt.figure(figsize=(8,8))

plt.subplot(2, 2, 1)
plt.imshow(np.log(PSF))
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(PSF)
plt.colorbar()

skew = 5

out = np.zeros(PSF[0, :pixel_shift+skew].shape)
for i in range(len(PSF)):
    out += PSF[i, :pixel_shift+skew]

plt.subplot(2, 2, 3)
# plt.plot(PSF[0, :pixel_shift])
plt.plot(out)

# plt.subplot(2, 2, 4)
# plt.plot(PSF[:pixel_shift, 0])

plt.show()