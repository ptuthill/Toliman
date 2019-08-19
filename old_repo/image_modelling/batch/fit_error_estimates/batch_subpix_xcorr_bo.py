#! python
# Add local scripts to module search path
import sys
import os
sys.path.append(os.path.realpath('../../toliman-proper'))
import numpy as np
from scipy.optimize import minimize
import json

import numpy as np
from proper_tools import form_detector_image, form_multi_psf, fix_prop_pixellate

#min_err = image_err(raw_image)

class Model:
    def __init__(self, target, as_to_m, detector_pitch, psf_ref, common_sampling,src_intensity_scale):
        self.target = target
        self.as_to_m = as_to_m
        self.x = [0.,0.,0.,0.,0.,0.]
        self.psf_a_pix = np.zeros_like(target)
        self.psf_b_pix = np.zeros_like(target)
        self.detector_pitch = detector_pitch
        self.common_sampling = common_sampling
        self.npsf = psf_ref.shape[0]
        self.npixels = target.shape[0]
        self.errors=[]
        self.psf_ref = psf_ref
        self.src_intensity_scale = src_intensity_scale

    def image_err(self,im):
        diff = im - self.target
        return np.sum((diff)**2)

    def model_image(self):
        return (self.psf_a_pix*self.x[2] + self.psf_b_pix*self.x[5]) * self.src_intensity_scale

    def set_psf_a(self,im):
        self.psf_a_pix = im
        return self

    def set_pos_a(self,dx,dy):
        self.x[0] = dx
        self.x[1] = dy
        self.set_psf_a(fix_prop_pixellate(self.psf_ref, self.common_sampling, self.detector_pitch, n_out = self.npixels, offset=(dx*self.as_to_m,dy*self.as_to_m)))

    def set_psf_b(self,im):
        self.psf_b_pix = im
        return self

    def set_pos_b(self,dx,dy):
        self.x[3] = dx
        self.x[4] = dy
        self.set_psf_b(fix_prop_pixellate(self.psf_ref, self.common_sampling, self.detector_pitch, n_out = self.npixels, offset=(dx*self.as_to_m,dy*self.as_to_m)))
        return self

    def set_major_flux(self,f):
        self.x[2] = f
        return self

    def set_minor_flux(self,f):
        self.x[5] = f
        return self

    def set_model_params(self,x):
        self.set_pos_a(x[0],x[1])
        self.set_major_flux(x[2])
        self.set_pos_b(x[3],x[4])
        self.set_minor_flux(x[5])
        return self

    def update_err(self,printerr=False, logerr=True):
        err = self.image_err(self.model_image())
        if logerr is True:
            self.errors.append((self.x.copy(),err))
        if printerr is True:
            print('Error for ', self.x, ': ', err)
        return err

    def model_err_major_pos(self,params, printerr=False, logerr=True):
        self.set_pos_a(params[0],params[1])
        return self.update_err(printerr, logerr)

    def model_err_minor_pos(self,params, printerr=False, logerr=True):
        self.set_pos_b(params[0],params[1])
        return self.update_err(printerr,logerr)

    def model_err_major_flux(self,f, printerr=False, logerr=True):
        self.set_major_flux(f)
        return self.update_err(printerr,logerr)

    def model_err_minor_flux(self,f, printerr=False, logerr=True):
        self.set_minor_flux(f)
        return self.update_err(printerr,logerr)

    def model_err(self, params, printerr=False, logerr=True):
        self.set_model_params(params)
        return self.update_err(printerr,logerr)

import numpy as np
from scipy import signal
from skimage.feature import peak_local_max
from skopt import gbrt_minimize,gp_minimize
#errors=[]

def fit_xcorr_bo(detector_image, detector_pitch, psf_ref, common_sampling, as_to_m, src_peak_photons,src_intensity_scale):
    # Assume npixels = 512
    model = Model(detector_image, as_to_m, detector_pitch, psf_ref, common_sampling,src_intensity_scale)

    # Use cross correlation to estimate major source location
    model.set_model_params([0,0,1.,0,0,0])
    psf_template=model.model_image()[256-50:256+50,256-50:256+50]
    psf_template -= psf_template.mean()

    corr = signal.correlate2d(detector_image, psf_template, boundary='symm', mode='same')
    max_loc = peak_local_max(corr,num_peaks=1)[0]

    # Estimate major peak
    est_flux = detector_image.max(axis=None)*1. / src_peak_photons
    model_bsf = [(max_loc[0] - 256)*detector_pitch/as_to_m, (max_loc[1] - 256)*detector_pitch/as_to_m, est_flux,0.,0.,0.]
    model.set_model_params(model_bsf)

    # Fine tune major position with BO
    params_init = [(-.5+model_bsf[0],.5+model_bsf[0]),(-.5+model_bsf[1],.5+model_bsf[1])]
    res = gbrt_minimize(model.model_err_major_pos, params_init)
    model_bsf[0] = res.x[0]
    model_bsf[1] = res.x[1]
    model.set_pos_a(model_bsf[0],model_bsf[1])

    # Tune major source flux with BO
    res = gp_minimize(model.model_err_major_flux, [(0.2,2.)])
    model.set_major_flux(res.x[0])
    model_bsf[2]=res.x[0]

    # Step 3: minor source position
    residual = detector_image - model.model_image()
    est_flux = residual.max(axis=None)*1. / src_peak_photons
    model.set_minor_flux(est_flux)
    model_bsf[5]=est_flux

    # Use cross correlation to estimate minor source location
    corr = signal.correlate2d(residual, psf_template, boundary='symm', mode='same')
    max_loc = peak_local_max(corr,num_peaks=1)[0]
    model_bsf[3] = (max_loc[0] - 256)*detector_pitch/as_to_m
    model_bsf[4] = (max_loc[1] - 256)*detector_pitch/as_to_m

    # Fine tune minor position with BO
    res = gbrt_minimize(model.model_err_minor_pos, [(-.5+model_bsf[3],.5+model_bsf[3]),(-.5+model_bsf[4],.5+model_bsf[4])])
    model_bsf[3] = res.x[0]
    model_bsf[4] = res.x[1]
    model.set_pos_b(res.x[0],res.x[1])

    # Final tweak
    params_init = [ (-.1+model_bsf[0],.1+model_bsf[0]),
                    (-.1+model_bsf[1],.1+model_bsf[1]),
                    (-.2+model_bsf[2],.2+model_bsf[2]),
                    (-.1+model_bsf[3],.1+model_bsf[3]),
                    (-.1+model_bsf[4],.1+model_bsf[4]),
                    (-.2+model_bsf[5],.2+model_bsf[5])]
    res = gbrt_minimize(model.model_err, params_init, x0=[model_bsf])
    return res


results_file = 'fit_results_xcorr_bo.txt'
nits = 30

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

import proper
detector_pitch = 11.0e-6 # m/pixel on detector
npixels = 512 # Size of detector, in pixels

# Calculated PSF
# First source, on axis
# Second source, offset 5" and fainter
from proper_tools import form_detector_image
def get_source(dx, dy, dI):
    settings = toliman_settings.copy()
    settings['tilt_x'] = dx
    settings['tilt_y'] = dy
    new  = {
            'wavelengths': [wavelength],
            'weights': [dI],
            'settings': settings
            }
    return new

params_a = (-1,-0.5,1.3) # Chosen at random
params_b = (params_a[0],params_a[1]+5.,params_a[2]/3.)
raw_image = form_detector_image('prescription_rc_quad', [get_source(params_a[0],params_a[1],params_a[2]), get_source(params_b[0],params_b[1],params_b[2])], gridsize, detector_pitch, npixels)
src_peak_photons = 25000
src_intensity_scale = (src_peak_photons/raw_image.max(axis=None))
raw_image = np.clip(src_intensity_scale*raw_image,0.,None)

psf_all = form_multi_psf('prescription_rc_quad', build_model_settings(0.,0.), gridsize, common_sampling, npixels*2)

def get_noisy_image(im):
    return np.random.poisson(np.clip(im,0.,None))

def measure_opt():
    #print('Error lower bound {}'.format(min_err))
#    opt = minimize(model_err, [0.1, 0.1,0.8], method='L-BFGS-B', bounds=[(-1.0,1.0),(-1.0,1.0),(0.5,1.5)])
    opt = fit_xcorr_bo(get_noisy_image(raw_image), detector_pitch, psf_all, common_sampling, as_to_m, src_peak_photons,src_intensity_scale)

    with open(results_file, 'a') as file:
        file.write(json.dumps({
                    'x': opt.x,
                    'err_bound': opt.fun
                    }))
        file.write('\n')

for i in range(nits):
    measure_opt()
