# Vary the tilt of one star and store the output
# Add local scripts to module search path
import sys
import os
import numpy as np
import pickle
sys.path.append(os.path.realpath('../../toliman-proper'))

from proper_tools import form_detector_image
import spirals

gridsize = 2048 # sampling of wavefront
detector_pitch = 11.0e-6 # m/pixel on detector
npixels = 512 # Size of detector, in pixels

# Define viewport for imaging:
vpmin = 256-64
vpmax = 256+64


toliman_settings = {
                    # Barnaby's values:
                    'diam': 0.001 * 2. * 150, 
                    'm1_fl': 571.7300 / 1000., #0.5717255, 
                    'm1_m2_sep': 549.240/1000., #0.54933763033373, 
                    'm2_fl': -23.3800/1000., # -0.02337895948640,  
                    'bfl': 590.000 / 1000., # 0.52761,#0.58999999998985,  
                    'm2_rad': 5.9 / 1000., #0.00590401477581,
#                    'm1_conic': -1.00011470000000,
#                    'm2_conic': -1.16799179177759,
                    'm2_strut_width': 0.01,
                    'm2_supports': 5,
                    'beam_ratio': 0.4,
                    'tilt_x': 0.00,
                    'tilt_y': 0.00,    
                    'phase_func': spirals.binarized_ringed
                    }


wl_gauss = [5.999989e-01,
            6.026560e-01,
            6.068356e-01,
            6.119202e-01,
            6.173624e-01,
            6.226281e-01,
            6.270944e-01,
            6.300010e-01 ]
weights_gaus = [5.3770e-02,
                1.1224e-01,
                1.5056e-01,
                1.7034e-01,
                1.7342e-01,
                1.5861e-01,
                1.2166e-01,
                5.9360e-02 ]


# First source, on axis
source_a = {
            'wavelengths': wl_gauss,
            'weights': [3.4*x for x in weights_gaus],
            'settings': toliman_settings
            }

# Second source, off-axis
tilted = toliman_settings.copy()

source_b = {
            'settings': tilted,
            'wavelengths': wl_gauss,
            'weights': weights_gaus
            }

delta_x = 1e-6
prescription ='prescription_rc_quad'
for i in range(100):
    tilted['tilt_x'] = 3.00 + i*delta_x
    tilted['tilt_y'] = 0.00
    fname = '{:09f}-{:09f}_{}.dump'.format(tilted['tilt_x'], tilted['tilt_y'], prescription)
    if not os.path.exists(fname):
        detector_image = form_detector_image(prescription, [source_a, source_b], gridsize, detector_pitch, npixels)
        with open(fname, 'wb') as outfile:
            pickle.dump(detector_image, outfile)
