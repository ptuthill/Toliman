import proper
import math
import numpy as np
import matplotlib.pyplot as plt
from prop_tilt import prop_tilt
from gen_opdmap import gen_opdmap
from build_phase_map import build_phase_map
from proper_cache import load_cacheable_grid

# from build_prop_circular_aperture import build_prop_circular_aperture
# prop conic

def prescription_quad(wavelength, gridsize, PASSVALUE = {}):
    # Assign parameters from PASSVALUE struct or use defaults
    diam           = PASSVALUE.get('diam',0.3)                    # telescope diameter in meters
    m1_fl          = PASSVALUE.get('m1_fl',0.5717255)             # primary focal length (m)
    beam_ratio     = PASSVALUE.get('beam_ratio',0.2)              # initial beam width/grid width
    tilt_x         = PASSVALUE.get('tilt_x',0.)                   # Tilt angle along x (arc seconds)
    tilt_y         = PASSVALUE.get('tilt_y',0.)                   # Tilt angle along y (arc seconds)
    noabs          = PASSVALUE.get('noabs',False)                 # Output complex amplitude?
    m1_hole_rad    = PASSVALUE.get('m1_hole_rad',None)            # Inner hole diameter
    use_caching    = PASSVALUE.get('use_caching',False)           # Use cached files if available?
    get_wf         = PASSVALUE.get('get_wf',False)                # Return wavefront

    """
    Prescription for a single quad lens system        
    """
    
    if 'phase_func' in PASSVALUE:
        print('DEPRECATED setting "phase_func": use "opd_func" instead')
        if 'opd_func' not in PASSVALUE:
            PASSVALUE['opd_func'] = PASSVALUE['phase_func']
    elif 'opd_func' not in PASSVALUE:
        print("no phase function")
    if 'phase_func_sec' in PASSVALUE:
        print('DEPRECATED setting "phase_func_sec": use "opd_func_sec" instead')
        if 'opd_func_sec' not in PASSVALUE:
            PASSVALUE['opd_func_sec'] = PASSVALUE['phase_func_sec']
    
    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)
    
    # Point off-axis
    prop_tilt(wfo, tilt_x, tilt_y)
    
    
    ###
    # Change to build ciruclar aperture??
    # Input aperture 
    proper.prop_circular_aperture(wfo, diam/2.)
    ###
    
    
    # Define entrance
    proper.prop_define_entrance(wfo)
    

    proper.prop_lens(wfo, m1_fl, "primary")

    if 'opd_func' in PASSVALUE:
        opd1_func = PASSVALUE['opd_func']        
        def build_m1_opd():
            return gen_opdmap(opd1_func, proper.prop_get_gridsize(wfo), proper.prop_get_sampling(wfo))

        wfo.wfarr *= build_phase_map(wfo, load_cacheable_grid(opd1_func.__name__, wfo, build_m1_opd, use_caching))

    if get_wf:
        wf = proper.prop_get_wavefront(wfo)
        print('Got wavefront')
        
        
    if m1_hole_rad is not None:
        proper.prop_circular_obscuration(wfo, m1_hole_rad)
        
    #if get_wf:
     #   wf = proper.prop_get_wavefront(wfo)
      #  print('Got wavefront')

    # Focus
    proper.prop_propagate(wfo, m1_fl, "focus", TO_PLANE=True)
        
    # End
    (wfo, sampling) = proper.prop_end(wfo)
    if get_wf:
        return (wfo, wf, sampling)
    else:
        return (wfo, sampling)
