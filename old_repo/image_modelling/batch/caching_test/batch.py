# Add local scripts to module search path
import sys
import os
sys.path.append(os.path.realpath('../../toliman-proper'))

from proper_cache import clear_all_cached

# Define some basics
from proper_tools import form_detector_image
from spirals import binarized_ringed_flipped

prescription = 'prescription_rc_quad'

beam_ratio = 0.4
gridsize = 2048

def binarized_ringed_650(r, phi):
    phase = 650.*1e-9*0.5
    return binarized_ringed_flipped(r, phi, phase)

toliman_settings = {
                    'diam': 0.001 * 2. * 150, 
                    'm1_fl': 571.7300 / 1000.,
                    'm1_m2_sep': 549.240/1000.,
                    'm2_fl': -23.3800/1000.,
                    'bfl': 590.000 / 1000., 
                    'm2_rad': 5.9 / 1000., 
                    'm2_strut_width': 0.01,
                    'm2_supports': 5,
                    'beam_ratio': beam_ratio,
                    'tilt_x': 0.00,
                    'tilt_y': 0.00,
                    'opd_func': binarized_ringed_650,
                    'm1_hole_rad':0.025,
                    'use_caching':True
                    }

detector_pitch = 11.0e-6 # m/pixel on detector
npixels = 512 # Size of detector, in pixels


# Single source, on axis, monochromatic
source_a = {
            'wavelengths': [0.6],
            'weights': [1.],
            'settings': toliman_settings
            }


import proper

def timed_op():
    return form_detector_image(prescription, [source_a], gridsize, detector_pitch, npixels) #, multi=False)

import timeit
# How many iterations to perform
nits = 1

results = {}

def doit():
#    return timeit.timeit('timed_op()', 'gc.enable()', number=nits, globals=globals())
    return timeit.timeit('timed_op()', number=nits, globals=globals())

clear_all_cached()
source_a['settings']['use_caching']=False
results['mono_uncached'] = doit()

clear_all_cached()
source_a['settings']['use_caching']=True
results['mono_savecached'] = doit()

source_a['settings']['use_caching']=True
results['mono_usecached'] = doit()

source_a['wavelengths'] = [5.999989e-01, 6.026560e-01, 6.068356e-01, 6.119202e-01, 6.173624e-01, 6.226281e-01, 6.270944e-01, 6.300010e-01 ]
source_a['weights'] = [5.3770e-02, 1.1224e-01, 1.5056e-01, 1.7034e-01, 1.7342e-01, 1.5861e-01, 1.2166e-01, 5.9360e-02 ]

clear_all_cached()
source_a['settings']['use_caching']=False
results['poly_uncached'] = doit()

clear_all_cached()
source_a['settings']['use_caching']=True
results['poly_savecached'] = doit()

source_a['settings']['use_caching']=True
results['poly_usecached'] = doit()

import json
import datetime
with open('caching_comparison_results.txt', 'a') as file:
#    file.write((datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S ")))
#    file.write(json.dumps({'nits': nits, 'gridsize':gridsize, 'beam_ratio':beam_ratio}))
    file.write(json.dumps(results))
#    file.write('\n')

print("Batch script done.")