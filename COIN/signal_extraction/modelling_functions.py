from scipy.interpolate import interp1d
import photometry
import numpy as np
from astropy.modeling.models import custom_model
from astropy.modeling import models,fitting
from scipy.special import jv
import matplotlib.pyplot as plt

def autocorrelate(im,larkin_transform=False,model_PSF=None,pad=True):
    
    if larkin_transform:
        im = photometry.larkin_transform(model_PSF,im)
        im = im**2
    # Should put padding here
    if pad:
        orig_shape = im.shape
        pad_im = photometry.pad_array(im)
    ft = np.fft.fft2(im)
    autocor = np.fft.ifft2(np.conj(ft)*ft)
    autocor = np.fft.fftshift(np.abs(autocor))
    if pad:
        autocor= photometry.depad_array(autocor,orig_shape)
    
    return autocor

@custom_model
def larkin_autocor_model(x,y,amplitude=1.,x_mean_0=0.,y_mean_0=0.,width=1.,background=0.):
    """ Custom astropy model for the autocorrelation of the Larkin transform. """
    intensity = 1./(2*np.pi) * (width/((x-x_mean_0)**2 + (y-y_mean_0)**2 + width**2)**1.5) # 2D Lorentzian works well
    intensity *= amplitude / np.max(intensity)
    intensity += background
    return intensity

@custom_model
def larkin_model(x,y,amplitude=1.,x_0=0.,y_0=0.,radius=1.,background = 0.):
    """ Custom astropy model for the Larkin Transform of an image. 
    Each peak should follow J1(R)/R, whereas the AiryDisk2D is (J1(R)/R)**2 .
    So this function uses sqrt(AiryDisk2D) plus a background term, and should be fit
    to the absolute value of the Larkin-Transformed image."""
    intensity = models.AiryDisk2D.evaluate(x,y,amplitude=amplitude,x_0=x_0,y_0=y_0,radius=radius)
    intensity = np.sqrt(np.abs(intensity))
    intensity += background
    return intensity

@custom_model
def larkin_model_negative(x,y,amplitude=1.,x_0=0.,y_0=0.,radius=1.,background = 0.):
    """ Custom astropy model for the Larkin Transform of an image. 
    Each peak should follow J1(R)/R. This function uses the actual Bessel function
    so that the correct values are negative, and should be fit to the Larkin-Transformed 
    image with no absolute value taken.
    """
    dist = np.hypot(x-x_0,y-y_0)/radius
    dist[dist<1e-7] += 1e-7 # stop divide by zero errors
    intensity = jv(1,dist) / dist
    intensity *= amplitude / intensity.max()
    intensity += background
    return intensity

def larkin_transform_fit(params,transformed_im,plot=True,fit_airy=False,rough_sep_pix = 45):
    """ Function to fit to the Larkin Transform of the image.
    params = [x0, y0, flux0, width0, sep, pa, flux1, width1]
    x0, y0 = position of primary relative to the central pixel
    flux0, flux1 = flux of primary and secondary
    width0,width1 = width of model for the primary and secondary
    sep,pa = separation and position angle of the binary
    
    transformed_im = larkin transformed image
    rough_sep_pix = approximate separation of the binary. Used for the fixed windowing, not the fit.
    
    plot: If set, will display a plot of the primary, model and residuals, and secondary, model and residuals
    fit_airy: If False (default) with fit sqrt(AiryDisk) to the data. If True, will fit AiryDisk.
    
    returns:
        [primary_fit,secondary_fit] astropy models
    """
    x0,y0,flux0,width0,sep,pa,flux1,width1 = params
    
    rough_sep_pix = int(round(rough_sep_pix))
    
    im_shape = transformed_im.shape
    y,x = np.indices(im_shape)
    y -= im_shape[0]//2
    x -= im_shape[1]//2
    
    x1 = x0 + sep*np.cos(pa*np.pi/180.)
    y1 = y0 + sep*np.sin(pa*np.pi/180.)

    fit = fitting.LevMarLSQFitter()
    
    if fit_airy:
        prim = models.AiryDisk2D(amplitude=flux0,x_0=x0,y_0=y0,radius=width0)
        sec = models.AiryDisk2D(amplitude=flux1,x_0=x1,y_0=y1,radius=width1)
    else:
        # Sqrt(AiryDisk), i.e. J1(r)/r
        prim = larkin_model(amplitude=flux0,x_0=x0,y_0=y0,radius=width0,background=0.)
        sec = larkin_model(amplitude=flux1,x_0=x1,y_0=y1,radius=width1,background=0.)

    
    # Fit the primary and secondary separately
    # Cut out a region
    cutsz = 6 # was 4
    cut_prim_im = transformed_im[im_shape[0]//2-cutsz:im_shape[0]//2+cutsz,
                                im_shape[1]//2-cutsz:im_shape[1]//2+cutsz]
    x_prim = x[im_shape[0]//2-cutsz:im_shape[0]//2+cutsz,
               im_shape[1]//2-cutsz:im_shape[1]//2+cutsz]
    y_prim = y[im_shape[0]//2-cutsz:im_shape[0]//2+cutsz,
               im_shape[1]//2-cutsz:im_shape[1]//2+cutsz]
    prim_fit = fit(prim,x_prim,y_prim,cut_prim_im,maxiter=100000,acc=1e-9)
    
    cut_sec_im = transformed_im[im_shape[0]//2-cutsz:im_shape[0]//2+cutsz,
                                im_shape[1]//2-cutsz+rough_sep_pix:im_shape[1]//2+cutsz+rough_sep_pix]
    x_sec = x[im_shape[0]//2-cutsz:im_shape[0]//2+cutsz,
               im_shape[1]//2-cutsz+rough_sep_pix:im_shape[1]//2+cutsz+rough_sep_pix]
    y_sec = y[im_shape[0]//2-cutsz:im_shape[0]//2+cutsz,
               im_shape[1]//2-cutsz+rough_sep_pix:im_shape[1]//2+cutsz+rough_sep_pix]
    sec_fit = fit(sec,x_sec,y_sec,cut_sec_im,maxiter=100000,acc=1e-9)

    
    if plot:
#         plt.imshow(model(x,y))
        m1 = prim_fit(x_prim,y_prim)
        m2 = sec_fit(x_sec,y_sec)
        
        plt.subplot(231)
        plt.imshow(cut_prim_im,origin='lower')
        plt.subplot(232)
        plt.imshow(m1,origin='lower')
        plt.subplot(233)
        plt.imshow(cut_prim_im-m1,origin='lower',vmin=-0.05,vmax=0.05)
        
        plt.subplot(234)
        plt.imshow(cut_sec_im,origin='lower')
        plt.subplot(235)
        plt.imshow(m2,origin='lower')
        plt.subplot(236)
        plt.imshow(cut_sec_im-m2,origin='lower',vmin=-0.05,vmax=0.05)
        print(prim_fit.parameters)
        print(sec_fit.parameters)
    
    sep = np.sqrt((prim_fit.x_0-sec_fit.x_0)**2 + (prim_fit.y_0-sec_fit.y_0)**2)

#     return sep
    return prim_fit,sec_fit

def autocorrelation_fit(im2, rough_sep = 44.93, rough_pa=0., cutout_sz=5, background_subtracted=False,
             plot_cutout=False, plot_resids=False):
    """ Fits to the distance of the second peak in the autocorrelation (i.e. the binary separation).
        This will perform a Levenberg-Marquardt fit over a small, fixed region around the peak. 
        The model used for the fit will depend on the options set.
        
        rough_sep, rough_pa: Defaults 44.93 pixels and 0 degrees
            These define the position of the region used for the fit.
        cutout_sz: Default 5 pixels
            The radius of the box used for the fit.
        background_subtracted: Default False
            If True, the model used for the fit will be an Airy Disk + constant background.
            If False, the model will be a Gaussian + planar background.
            These seemed to work best.
    """
    
    rough_pos = np.round([rough_sep*np.sin(rough_pa),rough_sep*np.cos(rough_pa)]).astype(int)
    calc_pos = [rough_sep*np.sin(rough_pa),rough_sep*np.cos(rough_pa)]
    
    cutout = np.copy(im2[im2.shape[0]//2+rough_pos[0]-cutout_sz:im2.shape[0]//2+rough_pos[0]+cutout_sz,
                 im2.shape[1]//2+rough_pos[1]-cutout_sz:im2.shape[1]//2+rough_pos[1]+cutout_sz])

    cutout /= np.max(cutout)
    if plot_cutout:
        plt.imshow(cutout)

    x,y = np.indices(cutout.shape)
    x = x + rough_pos[0] - cutout_sz
    y = y + rough_pos[1] - cutout_sz
    
    # Fit a Gaussian
    fit = fitting.LevMarLSQFitter()
    
    if background_subtracted:
        gauss = models.AiryDisk2D(amplitude=cutout.max(),x_0=calc_pos[0],y_0=calc_pos[1],radius=3.54)        
        bckgd = models.Const2D(amplitude=0.6)
    else:
        gauss = models.Gaussian2D(amplitude = cutout.max(),x_stddev=1.36,y_stddev=1.36,
                                  x_mean=calc_pos[0],y_mean=calc_pos[1])

        bckgd = models.Planar2D(slope_x = -0.00564816,slope_y=-0.02378304,intercept=1.01)
        gauss.fixed['theta']=True
        
    cutout_model = gauss + bckgd

    # fit the data with the fitter
    fitted_model = fit(cutout_model,x,y,cutout,maxiter=100000,acc=1e-7);
    
    # Rename the parameters so the output looks the same
    if background_subtracted:
        fitted_model.x_mean_0 = fitted_model.x_0_0
        fitted_model.y_mean_0 = fitted_model.y_0_0
        
    if plot_resids:
        test = fitted_model(x,y)
        plt.figure(figsize=(12,4))
        plt.clf()
        plt.subplot(131)
        plt.imshow(cutout,origin='lowerleft',vmin=0.7,vmax=1);plt.colorbar()
        plt.subplot(132)
        plt.imshow(test,origin='lowerleft',vmin=0.7,vmax=1);plt.colorbar()
        plt.subplot(133)
        plt.imshow(cutout-test,origin='lowerleft',vmin=-0.05,vmax=0.05);plt.colorbar()
    return fitted_model
