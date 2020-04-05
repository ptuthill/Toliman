import os
import sys
import numpy as np
from astropy.io import fits
import multiprocessing as mp
from FT_modelling import *

def run_model(inputs):
    """
    Runs the Broadband Fourier Model
    """
    x, wavels, weights, leakages, i, tf_size, path = inputs
    # Set parameters
    pupil = np.load("pupil_array.npy")
    aperture = 0.1
    m2_obsc = 0
    chip_dim = 512
    foc_length = 3.635
    pix_size = 6.5e-6
    y = 0
    transform_size = tf_size

    # Generate model
    im = FT_model_broadband(pupil, aperture, m2_obsc, chip_dim, wavels, weights, 
                            foc_length, pix_size, transform_size, x, y, polar=False, leakages=leakages)

    # Save numpy array
    npy_file_name = path + "/npy/{}.npy".format(i)
    np.save(npy_file_name, im)

    # Save fits file
    fits_file_name = path + "/fits/{}.fits".format(i)
    fits.writeto(fits_file_name, im)

    # Print which sim is being generated in order to give an idea of progress
    if i%10 != 0:
        print("{} ".format(i), end='')
    else:
        print("{} ".format(i), end='\n')

# Set precision value
delta_x = 0.000001
num_positions = 365
pixel_shift = 35
output_size = 256
cycles = 3
flux_ratio = 4
dark_noise = 20
num_wavels = 101 # always keep odd in order for leakages to work properly!

directory_name = "test_sim"
directory_path = os.getcwd()
photon_counts = 1e12

# photon_counts = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
# photon_counts_str = ["1e6", "1e7", "1e8", "1e9", "1e10", "1e11", "1e12", "1e13", "1e14", "1e15"]


delta_x_str = str(delta_x)
cores = 2 #10
transform_size = 2**8 #2**10

# Make directories and set paths
# path = os.getcwd() + "/data"
path = os.getcwd()
os.mkdir(path + "/data/{}".format(delta_x_str))
for dir_name in photon_counts_str:
    os.mkdir(path + "/{}/{}".format(delta_x_str, dir_name))

# Update path to subdirectory
path = path + "/{}".format(delta_x_str)
try:
    os.mkdir(path + "/npy")
    os.mkdir(path + "/fits")
except FileExistsError as e:
    pass

# Create wavels and leakages
wavels = np.linspace(545e-9, 645e-9, num=num_wavels)
# Split the leakage terms into two sections to more accurately represent the 
# leakage values provided in the ImagineOptix doccument
leakages = np.concatenate([np.linspace(0.02, 0.0275, num=3*(num_wavels-1)//10), 
                           np.linspace(0.0275, 0.0125, num=(7*(num_wavels-1)//10)+1)])

# Generate weights
mu = wavels[num_wavels//2]
sig = num_wavels//4
a = 1/(2*np.pi*sig)
b = ((wavels-mu/sig)**2)
power = a*np.exp(-0.5*b)
weights = power/np.max(power)

# Set positions
Ts = np.linspace(0, 2*np.pi, num=num_positions, endpoint=False)
Xs = delta_x*np.sin(Ts)

# Write signal file
with open(path + "/signal.csv", 'w') as f:
    f.write("t,x\n")
    for i in range(cycles*len(Ts)):
        f.write("{},{}\n".format(i, Xs[i%num_positions]))

# Generate inputs values for multithreading
input_vals = []
for i in range(len(Xs)):
    input_vals.append([Xs[i], wavels, weights, leakages, i, transform_size, path])

# Generate PSFs
pool = mp.Pool(processes=cores)
pool.map(run_model, input_vals)



#####

# Exit here and run combiner seperately
# sys.exit(1)

#####

# Scripts combined here




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







# Old script (that wasnt even run)


# # Get numpy file names
# file_names = np.array(os.listdir(path + "/npy/"))

# # Read in numpy files
# PSFs_in = []
# for i in range(len(file_names)):
#     PSFs_in.append(np.load(path + "/npy/{}".format(file_names[i])))

# # Get slice values
# c = PSFs_in[0].shape[0]//2
# s = output_size//2

# # Crop on axis PSF
# on_axis_PSF = PSFs_in[0][c-s:c+s, c-s:c+s]

# # Combine, scale and normalise PSFs
# PSFs_out = []
# for i in range(len(PSFs_in)):
#     PSF_out = np.zeros([2*s,2*s])
#     PSF_out += flux_ratio * on_axis_PSF
#     PSF_out += PSFs_in[i][c-s:c+s, c-s-pixel_shift:c+s-pixel_shift]
#     PSF_out = PSF_out/np.sum(PSF_out)
#     PSFs_out.append(PSF_out)

# # Makes a full cyclic series of PSFs
# PSFs = []
# for i in range(cycles):
#     for j in range(len(PSFs_out)):
#         PSFs.append(PSFs_out[j])

# # Generate final images by thowing photons and adding background noise
# for i in range(len(photon_counts)):
#     for j in range(len(PSFs)):
#         # Poission noise
#         photons = np.random.poisson(photon_counts[i]*PSFs[j])
#         # Gaussian noise
#         noise = np.abs(np.round(np.random.normal(scale=dark_noise, size=photons.shape)))
#         # Combine
#         image = photons + noise

#         # Save
#         file_name = path + "/{}/{}.fits".format(photon_counts_str[i], j)
#         fits.writeto(file_name, image, overwrite=True)
