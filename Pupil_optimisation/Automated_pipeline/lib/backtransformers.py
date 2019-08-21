"""
A script to contain all of the different functions used to backtransform a modifed wavefront into a coherent pupil

NOTE: Do NOT change any of the functions in here, simplycreate a new function and change the name (ie _1_1) to reflect the changes
    This is to ensure that we can always track how a pupil was generated and use the exact function that was used on a pupil
"""
from lib.secondary_functions import *
import numpy as np
from math import hypot
from numpy.fft import ifftshift, ifft2

def binary_backtransform_1_0(pupil_object, history, modified_WF):
    """
    Backtransforms given modified WF into a binary phase pupil
    Inputs:
        pupil_object, dictionary: Contains all the various values associated with the pupil as described in pupil_object.txt
        modified_WF, 2D complex array: The modfied wavefront array at the detector
        
        Change this
        meta_data: The modified meta_data dictionary containing the modification applied to the wavefront 
    
    
    Outputs:
        pupil_out, 2D complex array: The pupil binarised formed by the back transformation of the modified wavefront
        meta_data: The modified meta_data dictionary with the backtransform method apended
    Notes:
        Assumes square inputs
        Create a function to find the best threshold value
        Sometimes the threshold creates a unity phase pupil (potentially fixed by taking average of max and min values)
        
    """
    # Get values
    size = pupil_object.pupil_inputs["array_size"]
    sampling = pupil_object.pupil_inputs["sampling"]
    max_radius = pupil_object.pupil_inputs["max_radius"]
    min_radius = pupil_object.pupil_inputs["min_radius"]
    
    # Pad WF to correct size
    scale = 3.91 # THIS IS DETERMINED EXPERIMENTALLY AND MAY NEED TO BE CHANGED FOR VARYING INPUTS!!
    WF_pad = pad_array(modified_WF, int(round(size*scale)))
    
    # Inverse FT to get back pupil
    pupil_raw = ifft2(ifftshift(WF_pad))
    
    # Threshold about which binarisation is done
    threshold = (np.amax(np.abs(np.angle(pupil_raw))) + np.amin(np.abs(np.angle(pupil_raw))))/2
        
    pupil_out = np.zeros([size,size], dtype=complex)
    c = size//2
    for i in range(size):
        for j in range(size):
            r = hypot(i-c, j-c)*sampling
            if r > max_radius or r < min_radius: # Inner and outer limits of the pupil
                pupil_out[i][j] = np.complex(0,0)
            else:
                if np.abs(np.angle(pupil_raw[i][j])) >= threshold:
                    pupil_out[i][j] = np.complex(1,0)
                else:
                    pupil_out[i][j] = -np.complex(1,0)
                    
    # Append new item to history
    history.append("binary_backtransform_1_0, threshold = {}".format(threshold))
                    
    return pupil_out, history
    
    
def nonbinary_backtransform(pupil_object, modified_WF):
    """
    Backtransforms given modified WF into a nonbinary phase pupil
    Inputs:
        pupil_object, dictionary: Contains all the various values associated with the pupil as described in pupil_object.txt
        modified_WF, 2D complex array: The modfied wavefront array at the detector
    Outputs:
        pupil_out, 2D complex array: The pupil binarised formed by the back transformation of the modified wavefront
    Notes:
        Assumes square inputs
        Create a function to find the best threshold value
        
    """    
    # Get values
    size = pupil_object.pupil_inputs["array_size"]
    sampling = pupil_object.pupil_inputs["sampling"]
    max_radius = pupil_object.pupil_inputs["max_radius"]
    min_radius = pupil_object.pupil_inputs["min_radius"]
    
    # Pad WF to correct size
    scale = 3.91 # THIS IS DETERMINED EXPERIMENTALLY AND MAY NEED TO BE CHANGED FOR VARYING INPUTS!!
    WF_pad = pad_array(modified_WF, int(round(size*scale)))
    
    # Inverse FT to get back pupil
    pupil_raw = ifft2(ifftshift(WF_pad))
    
    pupil_out = np.zeros([size,size], dtype=complex)
    c = size//2
    for i in range(size):
        for j in range(size):
            r = hypot(i-c, j-c)*sampling
            if r > max_radius or r < min_radius: # Inner and outer limits of the pupil
                pupil_out[i][j] = np.complex(0,0)
            else:
                pupil_out[i][j] = np.exp(1j*np.angle(pupil_raw[i][j]))
                    
    return pupil_out