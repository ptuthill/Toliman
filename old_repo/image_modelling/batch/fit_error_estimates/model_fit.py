import numpy as np
from proper_tools import form_detector_image, form_multi_psf, fix_prop_pixellate

#min_err = image_err(raw_image)

class Model:
    def __init__(self, target, f_eff, detector_pitch, psf_ref, common_sampling):
        self.target = target
        #f_eff = 15. # metres
        self.as_to_m = f_eff * np.pi / (3600 * 180) #m/as
        self.x = [0.,0.,0.,0.,0.,0.]
        self.psf_a_pix = np.zeros_like(target)
        self.psf_b_pix = np.zeros_like(target)
        self.detector_pitch = detector_pitch
        self.common_sampling = common_sampling
        self.npsf = psf_ref.shape[0]
        self.npixels = target.shape[0]
        self.errors=[]

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
        err = image_err(model_image())
        if logerr is True:
            self.errors.append((self.x.copy(),err))
        if printerr is True:
            print('Error for ', self.x, ': ', err)
        return err

    def model_err_major_pos(self,params, printerr=False, logerr=True):
        set_pos_a(params[0],params[1])
        return update_err(printerr, logerr)

    def model_err_minor_pos(self,params, printerr=False, logerr=True):
        set_pos_b(params[0],params[1])
        return update_err(printerr,logerr)

    def model_err_major_flux(self,f, printerr=False, logerr=True):
        set_major_flux(f)
        return update_err(printerr,logerr)

    def model_err_minor_flux(self,f, printerr=False, logerr=True):
        set_minor_flux(f)
        return update_err(printerr,logerr)

    def model_err(params, printerr=False, logerr=True):
        set_model_params(params)
        return update_err(printerr,logerr)
