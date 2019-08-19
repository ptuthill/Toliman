import numpy as np
from scipy import signal
from skimage.feature import peak_local_max
from skopt import gbrt_minimize
#errors=[]

def fit_xcorr_bo(detector_image, detector_pitch, as_to_m, src_peak_photons):
    # Assume npixels = 512
    model = Model()
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
