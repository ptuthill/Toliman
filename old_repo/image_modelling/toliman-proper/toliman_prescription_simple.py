import proper
import math

# Adapted from proper/examples/hubble_simple.py

def toliman_prescription_simple(wavelength, gridsize):
    # Values from Eduardo's RC Toliman system
    diam = 0.3                               # telescope diameter in meters
    fl_pri = 0.5*1.143451                    # primary focal length (m)
    # BN 20180208
    d_pri_sec = 0.549337630333726  # primary to secondary separation (m)
#    d_pri_sec = 0.559337630333726            # primary to secondary separation (m)
    fl_sec = -0.5*0.0467579189727913         # secondary focal length (m)
    d_sec_to_focus = 0.528110658881 # nominal distance from secondary to focus (from eqn)
#    d_sec_to_focus = 0.589999999989853       # nominal distance from secondary to focus
    beam_ratio = 0.2                         # initial beam width/grid width

    m2_rad = 0.059 # Secondary half-diameter (m)
    m2_strut_width = 0.01 # Width of struts supporting M2 (m)
    m2_supports = 5

    # Define the wavefront
    wfo = proper.prop_begin(diam, wavelength, gridsize, beam_ratio)
    
    # Input aperture
    proper.prop_circular_aperture(wfo, diam/2)
    # NOTE: could prop_propagate() here if some baffling included
    # Secondary and structs obscuration
    proper.prop_circular_obscuration(wfo, m2_rad)                      # secondary mirror obscuration
    # Spider struts/vanes, arranged evenly radiating out from secondary
    strut_length = diam/2 - m2_rad
    strut_step = 360/m2_supports
    strut_centre = m2_rad + strut_length/2
    for i in range(0, m2_supports):
        angle = i*strut_step
        radians = math.radians(angle) 
        xoff = math.cos(radians)*strut_centre
        yoff = math.sin(radians)*strut_centre
        proper.prop_rectangular_obscuration(wfo, m2_strut_width, strut_length,
                                            xoff, yoff,
                                            ROTATION = angle + 90)
        
    # Define entrance
    proper.prop_define_entrance(wfo)

    # Primary mirror (treat as quadratic lens)
    proper.prop_lens(wfo, fl_pri, "primary")

    # Propagate the wavefront
    proper.prop_propagate(wfo, d_pri_sec, "secondary")

    # Secondary mirror (another quadratic lens)
    proper.prop_lens(wfo, fl_sec, "secondary")
    
    # NOTE: hole through primary?

    # Focus
    # BN 20180208 - Need TO_PLANE=True if you want an intermediate plane
    proper.prop_propagate(wfo, d_sec_to_focus, "focus", TO_PLANE=True)
#    proper.prop_propagate(wfo, d_sec_to_focus, "focus", TO_PLANE = False)

    # End
    (wfo, sampling) = proper.prop_end(wfo)

    return (wfo, sampling)
