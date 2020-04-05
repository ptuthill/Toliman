import os
import numpy as np
from astropy.io import fits
import multiprocessing as mp
from FT_modelling import *

def run_model(inputs):
    """
    Runs the Broadband Fourier Model
    """
    x, wavels, weights, leakages, i = inputs
    # Set parameters
    pupil = np.load("pupil_array.npy")
    aperture = 0.1
    m2_obsc = 0
    chip_dim = 600
    foc_length = 3.635
    pix_size = 6.5e-6
    y = 0
    transform_size = 2**11

    # Generate model
    im = FT_model_broadband(pupil, aperture, m2_obsc, chip_dim, wavels, weights, foc_length, pix_size, transform_size, x, y, polar=False, leakages=leakages)

    # Save numpy array
    npy_file_name = "Toliboy/full_cycle/{}/npy/{}.npy".format(delta_x, i)
    np.save(npy_file_name, im)

    # Save fits file
    fits_file_name = "Toliboy/full_cycle/{}/fits/{}.fits".format(delta_x, i)
    fits.writeto(fits_file_name, im)

    print(i)

# Set Wavels and Power
num_wavels = 101
wavels = np.linspace(545e-9, 645e-9, num=num_wavels)

mu = wavels[num_wavels//2]
sig = num_wavels//4
a = 1/(2*np.pi*sig)
b = ((wavels-mu/sig)**2)
power = a*np.exp(-0.5*b)
weights = power/np.max(power)
leakages = np.concatenate([np.linspace(0.02, 0.0275, num=30), np.linspace(0.0275, 0.0125, num=71)])

# Set positions
delta_x = 0.0001
Ts = np.linspace(0, 2*np.pi, num=365)
Xs = delta_x*np.sin(Ts)

# Write signal file
with open(os.getcwd() + "/Toliboy/full_cycle/{}/signal.csv".format(delta_x), 'w') as f:
    for i in range(3*len(Ts)):
        f.write("{},{}\n".format(i, Xs[i%365]))



input_vals = []
for i in range(len(Xs)):
    input_vals.append([Xs[i], wavels, weights, leakages, i])

pool = mp.Pool(processes=10)
pool.map(run_model, input_vals)

