import numpy as np
import proper

def normalise_sampling(wavefronts, samplings, common_sampling, npsf):
    """Resample each wavefront to a common grid
    
    Parameters
    ----------
    wavefronts : list of PROPER WaveFront class objects
        The wavefronts to be resampled
        
    samplings: list of floats 
        wavefront samplings for each wavefront, in metres
    
    common_sampling : float
        Target wavefront sampling in metres
        
    npsf : int
        Dimension of new image (npsf by npsf).
        
    Returns
    -------
    out : numpy ndarray
        Returns stack of wavefronts with common dimensions and sampling.
    """
    n = len(wavefronts)
    out = np.zeros([n, npsf, npsf], dtype = np.float64)
    # Resample and weight
    for i in range(n):
        wf = wavefronts[i] # np.abs(wavefronts[i])**2
        mag = samplings[i] / common_sampling
        out[i,:,:] = proper.prop_magnify(wf, mag, npsf, CONSERVE = True)
    return out

def combine_psfs(psfs, weights):
    """Combine stack of PSFs into a single 2D array 
    
    Parameters
    ----------
    psfs : 3D numpy array
        The PSFs to be combined, as a stack of 2D arrays of same dimensions
        
    weights : float
        Target wavefront sampling in meters
        
    Returns
    -------
    out : numpy ndarray
        Returns single 2D image array
    """
    out = psfs[0,:,:] * weights[0]    
    for i in range(1,len(weights)):
        out += psfs[i,:,:] * weights[i]
    return out

from numpy.fft import fft2, ifft2

# Looks like proper.prop_pixellate is broken, so paste directly and hack it up
def fix_prop_pixellate(image_in, sampling_in, sampling_out, n_out = 0, offset = None):
    """Integrate a sampled PSF onto detector pixels. 
    
    This routine takes as input a sampled PSF and integrates it over pixels of a 
    specified size. This is done by convolving the Fourier transform of the input 
    PSF with a sinc function representing the transfer function of an idealized 
    square pixel and transforming back. This result then represents the PSF 
    integrated onto detector-sized pixels with the same sampling as the PSF. 
    The result is interpolated to get detector-sized pixels at detector-pixel 
    spacing.
    
    Parameters
    ----------
    image_in : numpy ndarray
        2D floating image containing PSF
        
    sampling_in : float
        Sampling of image_in in meters/pixel
        
    sampling_out : float
        Size(=sampling) of detector pixels
        
    n_out : int
        Output image dimension (n_out by n_out)
        
    offset: tuple
        image offset in metres, as (dx,dy)
        
    Returns
    -------
    new : numpy ndarray
        Returns image integrated over square detector pixels.
    """
    n_in = image_in.shape[0]
    
    w = int(n_in/2)
    
    # Compute pixel transfer function (MTF)
    psize = 0.5 * (sampling_out / sampling_in)
    mag = sampling_in / sampling_out
    t = np.roll(np.arange(-w, w, dtype = np.float64), -w, 0) * psize * np.pi / w
#    t = np.arange(-w, w, dtype = np.float64) * psize * np.pi / w
    y = np.zeros(n_in, dtype = np.float64)
    y[1:] = np.sin(t[1:]) / t[1:]
    y[0] = 1.
    y = np.roll(y, +w, 0)
    
    pixel_mtf = np.dot(y[:,np.newaxis], y[np.newaxis,:])
    
    # Convolve image with detector pixel
    image_mtf = np.fft.fftshift(fft2(image_in)) 
    
    # shift image
    if offset is not None:
        (dx, dy) = offset
        vals = np.arange(-w, w)
        xoff,yoff = np.meshgrid(vals,vals)
        image_mtf*=np.exp(-1j*2*np.pi*(xoff*dx+yoff*dy)/(n_in*sampling_in))
        
    image_mtf /= image_in.size
    image_mtf *= pixel_mtf
    
    convolved_image = np.abs(ifft2(np.fft.ifftshift(image_mtf)) * image_mtf.size)
    image_mtf = 0
    convolved_image /= mag**2 #np.fft.ifftshift(convolved_image/mag**2)
    # Image is integrated over pixels but has original sampling; now, resample
    # pixel sampling
    if n_out == 0:
        n_out = int(np.fix(n_in * mag))
        
    new = proper.prop_magnify(convolved_image, mag, n_out)
    
    return new

def form_multi_psf(prescription, sources, gridsize, common_sampling, npsf, multi=True):
    source_psfs = []
    for source in sources:
        settings = source['settings']
        wavelengths = source['wavelengths']
        wl_weights = source['weights']

        if multi is True:
            (wavefronts, samplings) = proper.prop_run_multi(prescription, wavelengths, gridsize = gridsize, QUIET=True, PRINT_INTENSITY=False, PASSVALUE=settings)
            # prop_run_multi returns complex arrays, even when PSFs are intensity, so make real with abs
            psfs = normalise_sampling(np.abs(wavefronts), samplings, common_sampling, npsf)
        else:
            wavefronts = []
            samplings = []
            for wl in wavelengths:
                (wavefront, sampling) = proper.prop_run(prescription, wl, gridsize = gridsize, QUIET=True, PRINT_INTENSITY=False, PASSVALUE=settings)
                wavefronts.append(wavefront)
                samplings.append(sampling)
            psfs = normalise_sampling(wavefronts, samplings, common_sampling, npsf)
            
        source_psfs.append(combine_psfs(psfs, wl_weights))

    psf_all = combine_psfs(np.stack(source_psfs), [1. for i in range(len(source_psfs))])
    return psf_all

def form_detector_image(prescription, sources, gridsize, detector_pitch, npixels, multi=True, offset=None):
    common_sampling = detector_pitch/2. # for Nyquist
    npsf = npixels*2
    psf_all = form_multi_psf(prescription, sources, gridsize, common_sampling, npsf, multi=multi)
    return fix_prop_pixellate(psf_all, common_sampling, detector_pitch, offset=offset)
