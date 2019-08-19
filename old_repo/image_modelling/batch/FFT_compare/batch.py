# Add local scripts to module search path
import sys
import os
sys.path.append(os.path.realpath('../../toliman-proper'))

import timeit

import gc

# These parameters affect the sampling, and therefore the amount of FFT computation 
gridsize = 2048 # sampling of wavefront
beam_ratio = 0.4

# How many iterations to perform
nits = 1

results = {}

from proper_tools import form_detector_image
from spirals import binarized_ringed

prescription = 'prescription_rc_quad'
# The actual values probably don't matter all that much for the purposes of this comparison.

def binarized_ringed_650(r, phi):
    phase = 650.*1e-9*0.25
    return binarized_ringed(r, phi, phase)

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
                    'phase_func': binarized_ringed_650
                    }


detector_pitch = 11.0e-6 # m/pixel on detector
npixels = 512 # Size of detector, in pixels

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
            'wavelengths': wl_gauss[:4],
            'weights': weights_gaus[:4],
            'settings': toliman_settings
            }

# Second source, off-axis
tilted = toliman_settings.copy()
tilted['tilt_x'] = 3.00
tilted['tilt_y'] = 1.00

source_b = source_a.copy()
source_b['settings'] = tilted

def timed_op():
    return form_detector_image(prescription, [source_a, source_b], gridsize, detector_pitch, npixels) #, multi=False)

def doit():
#    return timeit.timeit('timed_op()', 'gc.enable()', number=nits, globals=globals())
    return timeit.timeit('timed_op()', number=nits, globals=globals())

import proper
proper.prop_use_ffti(DISABLE=True)
proper.prop_use_fftw(DISABLE=True)
results['numpy'] = doit()

proper.prop_use_fftw()
proper.prop_fftw_wisdom(gridsize)
results['fftw'] = doit()

proper.prop_use_ffti(MKL_DIR='/usr/physics/ic16/mkl/lib/intel64/')
results['intel'] = doit()

import json
import datetime
with open('fft_comparison_results.txt', 'a') as file:
    file.write((datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S ")))
    file.write(json.dumps({'nits': nits, 'gridsize':gridsize, 'beam_ratio':beam_ratio}))
    file.write(json.dumps(results))
    file.write('\n')