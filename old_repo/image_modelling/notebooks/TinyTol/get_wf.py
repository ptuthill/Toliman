import proper
import math
import numpy as np
#import matplotlib.pyplot as plt
from prop_tilt import prop_tilt
from gen_opdmap import gen_opdmap
from build_phase_map import build_phase_map
from proper_cache import load_cacheable_grid

def get_wf(wavelength, gridsize, PASSVALUE = {}):
    diam           = PASSVALUE.get('diam',0.3)                    # telescope diameter in meters
    m1_fl          = PASSVALUE.get('m1_fl',0.5717255)             # primary focal length (m)
    #beam_ratio     = PASSVALUE.get('beam_ratio',0.99)             # initial beam width/grid width
    tilt_x         = PASSVALUE.get('tilt_x',0.)                   # Tilt angle along x (arc seconds)
    tilt_y         = PASSVALUE.get('tilt_y',0.)                   # Tilt angle along y (arc seconds)

    beam_ratio = 0.99
    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)
    
    # Point off-axis
    prop_tilt(wfo, tilt_x, tilt_y)
    
    # Input aperture 
    proper.prop_circular_aperture(wfo, diam/2.)
    
    # Define entrance
    proper.prop_define_entrance(wfo)
    
    proper.prop_lens(wfo, m1_fl, "primary")

    opd1_func = PASSVALUE['opd_func']        
    def build_m1_opd():
        return gen_opdmap(opd1_func, proper.prop_get_gridsize(wfo), proper.prop_get_sampling(wfo))

    wfo.wfarr *= build_phase_map(wfo, load_cacheable_grid(opd1_func.__name__, wfo, build_m1_opd, False))

    wf = proper.prop_get_wavefront(wfo)
        
    return wf