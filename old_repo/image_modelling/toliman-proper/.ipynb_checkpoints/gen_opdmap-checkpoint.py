import math
import numpy as np

def gen_opdmap(opd_func, ngrid, sampling):
    """Generate the OPD map for a phase pupil
    
    Parameters
    ----------
    opd_func : function
        Function to calculate the OPD introduced at a given polar co-ordinate 
                
    ngrid : integer
        Size of (square) grid for wavefront
    
    sampling : float
        Sampling distance for grid, in metres
    
        
    Returns
    -------
    opd_map: nparray
        OPD map of optical path difference for each position of wavefront
    """
    
    opd_map = np.zeros([ngrid, ngrid], dtype = np.float64)
    c = ngrid/2.
    for i in range(ngrid):
        for j in range(ngrid):
            x = i - c
            y = j - c
            phi = math.atan2(y, x)
            r = sampling*math.hypot(x,y)
            opd_map[i][j] = opd_func(r, phi)
        
    return opd_map