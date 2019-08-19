import proper
import math
import numpy as np
from prop_conic import prop_conic
from prop_tilt import prop_tilt
from gen_opdmap import gen_opdmap
from build_prop_circular_aperture import build_prop_circular_aperture
from build_prop_circular_obscuration import build_prop_circular_obscuration
from build_prop_rectangular_obscuration import build_prop_rectangular_obscuration
from build_phase_map import build_phase_map
from proper_cache import load_cacheable_grid

def prescription_rc_quad(wavelength, gridsize, PASSVALUE = {}):
    # Assign parameters from PASSVALUE struct or use defaults
    diam           = PASSVALUE.get('diam',0.3)                    # telescope diameter in meters
    m1_fl          = PASSVALUE.get('m1_fl',0.5717255)             # primary focal length (m)
    m1_hole_rad    = PASSVALUE.get('m1_hole_rad',0.035)           # Radius of hole in primary (m)
    m1_m2_sep      = PASSVALUE.get('m1_m2_sep',0.549337630333726) # primary to secondary separation (m)
    m2_fl          = PASSVALUE.get('m2_fl',-0.023378959)          # secondary focal length (m)
    bfl            = PASSVALUE.get('bfl',0.528110658881)          # nominal distance from secondary to focus (m)
    beam_ratio     = PASSVALUE.get('beam_ratio',0.2)              # initial beam width/grid width
    m2_rad         = PASSVALUE.get('m2_rad',0.059)                # Secondary half-diameter (m)
    m2_strut_width = PASSVALUE.get('m2_strut_width',0.01)         # Width of struts supporting M2 (m)
    m2_supports    = PASSVALUE.get('m2_supports',5)               # Number of support structs (assumed equally spaced)
    tilt_x         = PASSVALUE.get('tilt_x',0.)                   # Tilt angle along x (arc seconds)
    tilt_y         = PASSVALUE.get('tilt_y',0.)                   # Tilt angle along y (arc seconds)
    noabs          = PASSVALUE.get('noabs',False)                 # Output complex amplitude?
    use_caching    = PASSVALUE.get('use_caching',False)           # Use cached files if available?
    get_wf         = PASSVALUE.get('get_wf',False)                # Return wavefront

    # Can also specify a opd_func function with signature opd_func(r, phi)
    if 'phase_func' in PASSVALUE:
        print('DEPRECATED setting "phase_func": use "opd_func" instead')
        if 'opd_func' not in PASSVALUE:
            PASSVALUE['opd_func'] = PASSVALUE['phase_func']
    if 'phase_func_sec' in PASSVALUE:
        print('DEPRECATED setting "phase_func_sec": use "opd_func_sec" instead')
        if 'opd_func_sec' not in PASSVALUE:
            PASSVALUE['opd_func_sec'] = PASSVALUE['phase_func_sec']
    
    
    
    def build_m2_obs():
        # Input aperture
        grid = build_prop_circular_aperture(wfo, diam/2)

        # Secondary and structs obscuration
        grid *= build_prop_circular_obscuration(wfo, m2_rad) # secondary mirror obscuration
        # Spider struts/vanes, arranged evenly radiating out from secondary
        strut_length = diam/2 - m2_rad
        strut_step = 360/m2_supports
        strut_centre = m2_rad + strut_length/2
        for i in range(0, m2_supports):
            angle = i*strut_step
            radians = math.radians(angle) 
            xoff = math.cos(radians)*strut_centre
            yoff = math.sin(radians)*strut_centre
            grid *= build_prop_rectangular_obscuration(wfo, m2_strut_width,                                     strut_length,
                                                xoff, yoff,
                                                ROTATION = angle + 90)
        return grid
    
    
    
    
    
    
    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)

    # Point off-axis
    prop_tilt(wfo, tilt_x, tilt_y)
    
    wfo.wfarr *= load_cacheable_grid('m2_obs', wfo, build_m2_obs, use_caching)
    
    # Normalize wavefront
    proper.prop_define_entrance(wfo)
    
    #if get_wf:
      #  wf = proper.prop_get_wavefront(wfo)
     #   print('Got wavefront')
    
    proper.prop_propagate(wfo, m1_m2_sep, "primary")
    
    # Primary mirror
    if 'opd_func' in PASSVALUE:
        opd1_func = PASSVALUE['opd_func']        
        def build_m1_opd():
            return gen_opdmap(opd1_func, proper.prop_get_gridsize(wfo), proper.prop_get_sampling(wfo))
        wfo.wfarr *= build_phase_map(wfo, load_cacheable_grid(opd1_func.__name__, wfo, build_m1_opd, use_caching))
        
    if get_wf:
        wf = proper.prop_get_wavefront(wfo)
        print('Got wavefront')
        
    if 'm1_conic' in PASSVALUE:
        prop_conic(wfo, m1_fl, PASSVALUE['m1_conic'], "conic primary")
    else:
        proper.prop_lens(wfo, m1_fl, "primary")
        
    wfo.wfarr *= build_prop_circular_obscuration(wfo, m1_hole_rad)

    # Secondary mirror
    proper.prop_propagate(wfo, m1_m2_sep, "secondary")
    if 'opd_func_sec' in PASSVALUE:
        opd2_func = PASSVALUE['opd_func_sec']
        def build_m2_opd():
            return gen_opdmap(opd2_func, proper.prop_get_gridsize(wfo), proper.prop_get_sampling(wfo))
        wfo.wfarr *= build_phase_map(wfo, load_cacheable_grid(opd2_func.__name__, wfo, build_m2_opd, use_caching))
        
    if 'm1_conic' in PASSVALUE:
        prop_conic(wfo, m2_fl, PASSVALUE['m2_conic'], "conic secondary")
    else:
        proper.prop_lens(wfo, m2_fl, "secondary")
                
    def build_m2_ap():
        return build_prop_circular_aperture(wfo, m2_rad)
    wfo.wfarr *= load_cacheable_grid('m2_ap', wfo, build_m2_ap)

#    proper.prop_state(wfo)

    # Hole through primary
    if m1_m2_sep<bfl:
        proper.prop_propagate(wfo, m1_m2_sep, "M1 hole")
        def build_m1_hole():
            return build_prop_circular_aperture(wfo, m1_hole_rad) 
        wfo.wfarr *= load_cacheable_grid('m1_hole', wfo, build_m1_hole)


    # Focus - bfl can be varied between runs
    if m1_m2_sep<bfl:
        proper.prop_propagate(wfo, bfl-m1_m2_sep, "focus", TO_PLANE=True)
    else:
        proper.prop_propagate(wfo, bfl, "focus", TO_PLANE=True)

#     # End
#     return proper.prop_end(wfo, NOABS = noabs)

    # End
    (wfo, sampling) = proper.prop_end(wfo)
    if get_wf:
        return (wfo, wf, sampling)
    else:
        return (wfo, sampling)
