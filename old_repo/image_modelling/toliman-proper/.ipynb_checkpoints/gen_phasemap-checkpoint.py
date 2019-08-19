import math
import numpy as np
from gen_opdmap import gen_opdmap

def gen_phasemap(phase_func, ngrid, sampling, use_cached=True, save_cached=True):
    """Generate the phase map for a phase pupil
    
    Parameters
    ----------
    phase_func : function
        Function to calculate the phase shift at a given polar co-ordinate 
                
    ngrid : integer
        Size of (square) grid for wavefront
    
    sampling : float
        Sampling distance for grid, in metres
    
        
    Returns
    -------
    phase_map: nparray
        Phase map of optical path difference for each position of wavefront
    """
    print("WARNING: gen_phasemap DEPRECATED, use gen_opdmap instead")
    return gen_opdmap(phase_func, ngrid, sampling, use_cached=use_cached, save_cached=save_cached)