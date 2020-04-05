import numpy as np
import json
from lib.conversions import *
from lib.FT_modelling import *

def evaluate_model(inputs):
    """
    Evaluates the model using Least squares
    """
    data = np.load("zmx_zoom.npy")
    data_norm = data/np.sum(data)
    
    model = build_model_cartesian(inputs)
    model_norm = model/np.sum(model)
    
    # Take the sum of the squares of the different
    diff = model_norm - data_norm
    squares = np.square(diff)
    sum_sq = np.sum(squares)
    
    return sum_sq

def evaluate_model_shift(inputs):
    """
    Evaluates the model using Least squares
    """
    data = np.load("zmx_shift.npy")
    data_norm = data/np.sum(data)
    
    model = build_model_cartesian(inputs)
    model_norm = model/np.sum(model)
    
    # Take the sum of the squares of the different
    diff = model_norm - data_norm
    squares = np.square(diff)
    sum_sq = np.sum(squares)
    
    return sum_sq

def build_model_polar(inputs):
    """
    Models the PSF using a Fourier Transform
    
    Inputs: [r, phi] coordinate offset of obervered star from normal of telescope 
        r - arcseconds
        phi - radians
        
    Returns:
        Numpy array of the PSF
    """
    # Extract input parameters
    r, phi = inputs
    
    # Get simulation values
    with open("simulation_values.txt") as f:
        sim_vals = json.load(f)
        
    # Get pupil array
    pupil = np.load("pupil")
    
    PSF = FT_model(pupil, sim_vals["aperture"], sim_vals["m2_obsc"], sim_vals["detector_size"], sim_vals["wavelength"], sim_vals["focal_length"], 
                   sim_vals["detector_pitch"], sim_vals["transform_size"], r, phi, polar=True)
    
    return PSF

def build_model_cartesian(inputs):
    """
    Models the PSF using a Fourier Transform
    
    Inputs: [x y] offset from centre of array
        x - pixels
        y - pixels
        
    Returns:
        Numpy array of the PSF
    """
    # Extract input parameters
    x, y = inputs
    
    # Get simulation values
    with open("simulation_values.txt") as f:
        sim_vals = json.load(f)
        
    # Get pupil array
    pupil = np.load("pupil_array.npy")
    
    PSF = FT_model(pupil, sim_vals["aperture"], sim_vals["m2_obsc"], sim_vals["detector_size"], sim_vals["wavelength"], sim_vals["focal_length"], 
                   sim_vals["detector_pitch"], sim_vals["transform_size"], x, y, polar=False)
    
    return PSF

