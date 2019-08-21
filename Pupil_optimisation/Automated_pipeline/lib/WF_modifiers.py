"""
A script to hold all the different functions used to modify the wavefront

NOTE: Do NOT change any of the functions in here, simply create a new function and change the name (ie _1_1) to reflect the changes
    This is to ensure that we can always track how a pupil was generated and use the exact function that was used on a pupil
"""
from lib.secondary_functions import *
import numpy as np
from math import hypot
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from copy import deepcopy

def G_Saxburg_1_0(pupil_object, inputs_dictionary): 
    """
    Applies a single G-Saxburg iteration to the wavefront
    Inputs: 
        pupil_object, dictionary: Contains all the various values associated with the pupil as described in pupil_object.txt
        outer_limit, fringes: Radial distance at which the wavefront is gievn zero power outwards from 
        inner_limit, fringes: Radial distance at which the wavefront is gievn zero power inwards from 
    Output:
        WF_out, 2D complex array: Array of wavefront at the detctor after modifiation
    Note: 
        This function does not return the backtransformed pupil from the modified array
        Assumes a square input
    """
    # Unpack input objects
    outer_limit = inputs_dictionary["outer_limit"]
    inner_limit = inputs_dictionary["inner_limit"]
    
    # Get values
    ap = pupil_object.simulation_settings["aperture"]
    fl = pupil_object.simulation_settings["focal_length"]
    wl = pupil_object.simulation_settings["wavelength"]
    dp = pupil_object.simulation_settings["detector_pitch"]
    outer_limit_pix = fringes_to_pixels(outer_limit, ap, fl, wl, dp)
    inner_limit_pix = fringes_to_pixels(inner_limit, ap, fl, wl, dp)
    
    # Construct new WF object
    WF = pupil_object.images["WF"]
    size = WF.shape[0]
    c = size//2
    WF_out = np.zeros([size, size], dtype=complex)
    
    # Modify new WF object
    for i in range(size):
        for j in range(size):
            r = hypot(i-c, j-c)
            if r > outer_limit_pix or r < inner_limit_pix:
                mag = 0
            else:
                mag = 1
            phi = np.angle(WF[i][j])
            WF_out[i][j] = mag*np.exp(1j*phi)
    
    # Add the modification to the history
    history = deepcopy(pupil_object.history)
    history.append("G-Saxburg_1_0({}, {})".format(outer_limit, inner_limit))
                    
    return WF_out, history

