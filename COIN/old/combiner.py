import os
import numpy as np
from astropy.io import fits
import multiprocessing as mp
from FT_modelling import *

# Set precision value
delta_x = 0.0001
path = os.getcwd() + "/Toliboy/full_cycle/{}/npy/".format(delta_x)
save_path = os.getcwd() + "/Toliboy/full_cycle/{}/".format(delta_x)
file_names = np.array(os.listdir(path))

# Read in numpy files
images = []
for i in range(len(file_names)):
    im = np.load(path+file_names[i])
    images.append(im)

# Take on axis image
on_axis_full = images[0]
c = on_axis_full.shape[0]//2

# Set shift and size values
pixel_shift = 35
s = 400//2

# Combine PSFs
psfs = []
for i in range(len(images)):
    psf_out = np.zeros([2*s,2*s])
    psf_out += 4 * on_axis_full[c-s:c+s, c-s:c+s]
    psf_out += images[i][c-s:c+s, c-s-pixel_shift:c+s-pixel_shift]
    psf_out = psf_out/np.sum(psf_out)
    psfs.append(psf_out)

# Makes a full triple series of PSFs
final_psfs = []
for i in range(3):
    for j in range(len(psfs)):
        final_psfs.append(psfs[j])

# Throw photons at the PSFs
photon_counts = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
photon_counts_str = ["1e4", "1e5", "1e6", "1e7", "1e8", "1e9"]
for i in range(len(photon_counts)):
    for j in range(len(final_psfs)):

        photons = np.random.poisson(photon_counts[i]*final_psfs[j])
        noise = np.round(np.random.normal(scale=20, size=photons.shape))
        image = photons+noise

        # Save the files
        file_name = "{}/{}.fits".format(photon_counts_str[i], j)
        fits.writeto(save_path + file_name, image, overwrite=True)



