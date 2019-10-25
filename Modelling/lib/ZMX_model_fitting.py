import numpy as np
import json
from scipy import ndimage
from lib.conversions import *

def evaluate_model_zmx(inputs):
    """
    Evaluates the model using Least squares
    """
    data = np.load("zmx_noisy.npy")
    data_norm = data/np.sum(data)
    
    model = build_model_zmx(inputs)
    model_norm = model/np.sum(model)
    
    # Take the sum of the squares of the different
    diff = model_norm - data_norm
    squares = np.square(diff)
    sum_sq = np.sum(squares)
    
    return sum_sq

def build_model_zmx(inputs):
    """
    Models the PSF with Zemax
    Currently uses third order spline interpolation
    
    Inputs: [x, y, photons] offset from centre of array
        x - pixels
        y - pixels
        
    Returns:
        Numpy array of the PSF
        
    Notes:
        Sometimes returns small negatvie values (< -1e-10) - needs testing.
        Likely could be from third order spline interpolation
    """
    # Extract input parameters
    x, y = inputs
    
    # Get simulation values
    with open("simulation_values.txt") as f:
        sim_vals = json.load(f)
        
    # Get raw psf
    zmx_im = np.load("zmx_im.npy")
            
    # Shift psf
    size_ratio = sim_vals["zmx_pixel_size"]/sim_vals["detector_pitch"]
    shift = [x/size_ratio, y/size_ratio]
    
    zmx_shift = ndimage.shift(zmx_im, shift, order=sim_vals["interp_order"])
    
    
    # Downsample and normalise to detector
    im_shift = ndimage.zoom(zmx_shift, size_ratio)
    PSF = im_shift/np.sum(im_shift)
    
    return PSF