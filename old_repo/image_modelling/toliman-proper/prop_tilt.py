import proper
import numpy as np

def prop_tilt(wf, tilt_x, tilt_y):
    """Tilt a wavefront in X and Y.
    
    based on tilt(self, Xangle, Yangle) from Poppy (poppy_core.py)
    
    From that function's docs:
    The sign convention is chosen such that positive Yangle tilts move the star upwards in the
    array at the focal plane. (This is sort of an inverse of what physically happens in the propagation
    to or through focus, but we're ignoring that here and trying to just work in sky coords)
    
    Parameters
    ----------
    wf : obj
        WaveFront class object
                
    tilt_x : float
        Tilt angle along x in arc seconds
    
    tilt_y : float
        Tilt angle along y in arc seconds
        
    Returns
    -------
        None
        Modifies wavefront array in wf object.
    """

    
    if np.abs(tilt_x) > 0 or np.abs(tilt_y) > 0:
        sampling = proper.prop_get_sampling(wf) # m/pixel        
        xangle_rad = tilt_x *  np.pi / 648000. # rad.
        yangle_rad = tilt_y *  np.pi / 648000. # rad.
        
        ngrid = proper.prop_get_gridsize(wf) # pixels
        U, V = np.indices(wf.wfarr.shape, dtype=float)
        U -= (ngrid - 1) / 2.0 # pixels X
        U *= sampling # m
        V -= (ngrid - 1) / 2.0 # pixels Y
        V *= sampling # m
        
        # Not totally comfortable that these are combined linearly 
        # but go with Poppy for now
        phase = V * xangle_rad + U * yangle_rad  # rad. m
        proper.prop_add_phase(wf, phase)
        #tiltphasor = np.exp(2.0j * np.pi/wf.lamda * phase )
        #wf.wfarr *= tiltphasor