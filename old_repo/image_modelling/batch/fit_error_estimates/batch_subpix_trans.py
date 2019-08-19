#! python
# Add local scripts to module search path
import sys
import os
sys.path.append(os.path.realpath('../../toliman-proper'))
import numpy as np
from scipy.optimize import minimize
import json

results_file = 'fit_results_subpix.txt'
nits = 30
peak_photons = 25000

from spirals import binarized_ringed_flipped
def binarized_ringed_650(r, phi):
    phase = 650.*1e-9*0.5
    return binarized_ringed_flipped(r, phi, phase)

beam_ratio = 0.4
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

gridsize = 2048 # sampling of wavefront
wavelength = 0.65 # microns
detector_pitch = 11.0e-6 # m/pixel on detector
npixels = 512 # Size of detector, in pixels
common_sampling = detector_pitch/2. # for Nyquist 
f_eff = 15. # metres
as_to_m = f_eff * np.pi / (3600 * 180) #m/as
# should be 7.2722052166430399038487115353692e-5 # m/as

def build_model_settings(tiltx,tilty):
    settings = toliman_settings.copy()
    settings['tilt_x'] = tiltx 
    settings['tilt_y'] = tilty
    source = {
            'wavelengths': [wavelength],
            'weights': [1.],
            'settings': settings
            }
    return [source]

# Calculated PSF using on-axis image
from proper_tools import form_detector_image, form_multi_psf, fix_prop_pixellate

def get_synth_image(tiltx,tilty,flux):
    sources = build_model_settings(tiltx,tilty)
    im = form_detector_image('prescription_rc_quad', sources, gridsize, detector_pitch, npixels)
    # scale it to decent number of photons
    peak_photons = 25000 * flux
    photon_scale = peak_photons/im.max(axis=None)
    im *= photon_scale
    return im

raw_image = get_synth_image(0.,0.,1.)

psf_all = form_multi_psf('prescription_rc_quad', build_model_settings(0.,0.), gridsize, common_sampling, npixels*2)

def get_tilted_image(tiltx,tilty,flux):
    im = fix_prop_pixellate(psf_all, common_sampling, detector_pitch, offset=(tiltx*as_to_m,tilty*as_to_m),n_out = npixels)
    # scale it to decent number of photons
    peak_photons = 25000 * flux
    photon_scale = peak_photons/im.max(axis=None)
    im *= photon_scale
    return im


def get_noisy_image(im):
    return np.random.poisson(np.clip(im,0.,None))

def measure_opt():
    source_image = get_noisy_image(raw_image)
    
    def image_err(im):
        diff = im - source_image
        return np.sum((diff)**2)
    
    min_err = image_err(raw_image)
    
    def model_err(pos):
        tiltx = pos[0]
        tilty = pos[1]
        flux = pos[2]        
        im = get_tilted_image(tiltx,tilty,flux)
        err = image_err(im)
#        print("Error for ({},{},{}): {} ({}%)".format(tiltx,tilty,flux,err,100.*err/min_err))
        return err
    
    #print('Error lower bound {}'.format(min_err))
    opt = minimize(model_err, [0.1, 0.1,0.8], method='L-BFGS-B', bounds=[(-1.0,1.0),(-1.0,1.0),(0.5,1.5)])
    with open(results_file, 'a') as file:
        file.write(json.dumps({
                    'fun': opt.fun,
                    'x': opt.x.tolist(),
                    'nfev': opt.nfev,
                    'status': opt.status,
                    'err_bound': min_err
                    }))
        file.write('\n')

for i in range(nits):
    measure_opt()
