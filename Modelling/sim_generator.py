import numpy as np
from astropy.io import fits
import multiprocessing as mp
from lib.FT_modelling import *

# Set perturbed paramters
wavels = np.linspace(456e-9, 556e-9, num=11)
leakages = np.concatenate([np.linspace(0.02, 0.0275, num=3), np.linspace(0.0275, 0.0125, num=8)])
positions = np.linspace(-0.5, 0.5, num=5)

def run_model(inputs):
    
    x, y, wavel, leakage = inputs
    # Set science parameters
    pupil = np.load("pupil_array.npy")
    aperture = 0.018
    m2_obsc = 0
    chip_dim = 128
    foc_length = 1.37
    pix_size = 7.4e-6
    transform_size = 2**8 # Subject to change - test first

    im = FT_model(pupil, aperture, m2_obsc, chip_dim, wavel, foc_length, pix_size, transform_size, x, y, polar=False, leakage=leakage)
            
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = chip_dim
    header["NAXIS2"] = chip_dim
    header["XPOS"] = x
    header["YPOS"] = y
    header["WAVEL"] = wavel
    header["LEAK"] = leakage
            
    file_name = "single_psf/{}_{}_{}.fits".format(x, y, np.round(wavel*1e9))
    fits.writeto(file_name, im, header=header)

input_vals = []
for x in positions:
    for y in positions:
        for wavel, leakage in zip(wavels, leakages):
            
            input_vals.append([np.round(x, decimals=2), np.round(y, decimals=2), wavel, leakage])
            
pool = mp.Pool(processes=4)
pool.map(run_model, input_vals)

