import os
import numpy as np
from astropy.io import fits

# Set precision value
delta_x = 0.000001
pixel_shift = 35
output_size = 256
cycles = 3
flux_ratio = 1
dark_noise = 20
photon_counts = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
photon_counts_str = ["1e6", "1e7", "1e8", "1e9", "1e10", "1e11", "1e12", "1e13", "1e14", "1e15"]

# Generate paths and get files
path = os.getcwd() + "/data/{}".format(delta_x)
file_names = np.array(os.listdir(path + "/npy/"))

# Read in numpy files
PSFs_in = []
for i in range(len(file_names)):
    PSFs_in.append(np.load(path + "/npy/{}".format(file_names[i])))

# Get slice values
c = PSFs_in[0].shape[0]//2
s = output_size//2

# Crop on axis PSF
on_axis_PSF = PSFs_in[0][c-s:c+s, c-s:c+s]

# Combine, scale and normalise PSFs
PSFs_out = []
for i in range(len(PSFs_in)):
    PSF_out = np.zeros([2*s,2*s])
    PSF_out += flux_ratio * on_axis_PSF
    PSF_out += PSFs_in[i][c-s:c+s, c-s-pixel_shift:c+s-pixel_shift]
    PSF_out = PSF_out/np.sum(PSF_out)
    PSFs_out.append(PSF_out)

# Makes a full triple series of PSFs
PSFs = []
for i in range(cycles):
    for j in range(len(PSFs_out)):
        PSFs.append(PSFs_out[j])

# Create new directories
if flux_ratio == 1:
    try:
        os.mkdir(path + "/equal_flux/")
    except FileExistsError as e:
        pass
    path = path + "/equal_flux"

for i in range(len(photon_counts_str)):
    try:
        os.mkdir(path + "/{}/".format(photon_counts_str[i]))
    except FileExistsError as e:
        pass

# Throw photons at the PSFs
for i in range(len(photon_counts)):
    for j in range(len(PSFs)):
        # Poisson noise
        photons = np.random.poisson(photon_counts[i] * PSFs[j])
        # Gaussian noise
        noise = np.round(np.random.normal(scale=dark_noise, size=photons.shape))
        # Combine
        image = photons + noise

        # Save the files
        file_name = path + "/{}/{}.fits".format(photon_counts_str[i], j)
        fits.writeto(file_name, image, overwrite=True)
