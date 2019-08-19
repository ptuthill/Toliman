import proper
import math
from prop_tilt import prop_tilt

def prescription_quad_tiltafter(wavelength, gridsize, 
                          PASSVALUE = {'diam': 0.3, 
                                       'm1_fl': 0.5717255, 
                                       'beam_ratio': 0.2, 
                                       'tilt_x': 0.0,
                                       'tilt_y': 0.0
                                      }):
    diam           = PASSVALUE['diam']           # telescope diameter in meters
    m1_fl          = PASSVALUE['m1_fl']          # primary focal length (m)
    beam_ratio     = PASSVALUE['beam_ratio']     # initial beam width/grid width
        
    tilt_x         = PASSVALUE['tilt_x']         # Tilt angle along x (arc seconds)
    tilt_y         = PASSVALUE['tilt_y']         # Tilt angle along y (arc seconds)
    
    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)
        
    # Input aperture
    proper.prop_circular_aperture(wfo, diam/2)
    
    # Define entrance
    proper.prop_define_entrance(wfo)

    # Primary mirror (treat as quadratic lens)
    proper.prop_lens(wfo, m1_fl, "primary")

    # Point off-axis
    prop_tilt(wfo, tilt_x, tilt_y)

    # Focus
    proper.prop_propagate(wfo, m1_fl, "focus", TO_PLANE=True)

    # End
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
