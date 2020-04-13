import numpy as np
from lib.FT_modelling import *

def clean(im, m, ma_val, margs, shave=0.05):
    """
    im: data image
    m: model
    ma_val: model auto correlation value
    
    """
    # Get correlation map and values
    cmap = correlate(im, m)
    p, peak = get_max(cmap)
    
    # Shave and flux values
    flux = shave*peak*ma_val
    
    # Generate psf
    pupil, aperture, m2_obsc, npix, wl, fl, pix_size, tf_size = margs
    PSF = flux*FT_model(pupil, aperture, m2_obsc, npix, wl, fl, pix_size, 
                                  tf_size, p[0], p[1], polar=False)
    
    # Generate output image and return object
    im_out = im - PSF    
    return im_out, p, flux, PSF

def reconstruct_stars(positions, fluxes):
    """
    Takes the positions and fluxes and groups them by common positions, 
    summing the fluxes
    """
    stars = []
    fluxes = np.array(fluxes)
        
    # Get unique position values
    upos, idxs = np.unique(positions, axis=0, return_inverse=True) 
    for i in range(len(upos)):
        pos = upos[i]
        
        temp = fluxes
        vals = np.multiply(temp, idxs==i)
        flux = np.sum(vals)
        
        stars.append([pos, flux])
    return stars

def correlate(im1, im2):
    """
    Returns the correlation of two images using the convolution theorem
    This automatically pads and unpads in the process
    """
    # Pad 
    im1_p, shape = pad_array(im1)
    im2_p, shape = pad_array(im2)
    
    # Transform
    FT_im1 = np.fft.fft2(im1)
    FT_im2 = np.fft.fft2(im2)
    
    # Mutliply and inverse tranform
    product = np.multiply(FT_im1, FT_im2)
    compl_corre_map = np.fft.fftshift(np.fft.ifft2(product))
    
    # Take real
    corre_map = np.real(compl_corre_map)
    
    # Return de-padded array
    return depad_array(corre_map, shape)

def convolve(arr1, arr2, pad=False):
    """
    Convolves two arrays with each other using the convolution theorem
    if pad == True: Function will pad the arrays
    else: Assumes padding already has been done
    """
    # Pad array
    if pad:
        arr1, shape = pad_array(arr1)
        arr2, shape = pad_array(arr2)
        
    FT1 = np.fft.fft2(arr1)
    FT2 = np.fft.fft2(arr2)
    mult = np.multiply(FT1, FT2)
    conv = np.fft.fftshift(np.fft.ifft2(mult))
    
    # Depad array
    if pad:
        conv = depad_array(conv, shape)
        
    return conv
    
def get_max(corr_map, raw=False):
    """
    Takes in the correlation map and returns the position of the peak correlation
    and the the correlation strength value
    Inputs: 
        corr_map, The cross correlation map
        raw: When True, returns pos in the coordinate system of the Fourier modelling code
    Returns:
        pos, The position as defined in the coordiante system fed into the FT code
        peak, The peak correlation value, used to reconstruct the flux of the PSF

    """
    peak = np.max(corr_map)
    idx = np.array(np.where(corr_map == peak)).reshape(2)
    if not raw:
        pos = idx_to_pos(idx, corr_map.shape[0])
        return pos, peak
    return idx, peak

def pad_array(array):
    """
    Pads the input array by a factor of 2
    
    Returns:
        array_out - The padded array in numpy format
        shape_in - The input shape of the array used to de-pad the array later
    
    """
    # Ensure we have a numpy array format
    array = np.array(array)
    shape_in = np.array(array.shape)
    array_out = np.zeros(2*shape_in, dtype=array.dtype)
    array_out[:shape_in[0], :shape_in[1]] = array
    
    return array_out, shape_in

def depad_array(array, shape_out):
    """
    De-pads the array to the dimension of shape_out
    the shape_out parameter should be fed in as the returned "shape_in" 
    parameter from the pad_array function
    """
    # Ensure we have a numpy array format
    i, j = shape_out
    return array[:i, :j]

def idx_to_pos(p, npix):
    """
    Takes in the index returned from the correlation images and converts
    the values into the positons fed into the FT code
    """
    return np.array([p[1] - (npix//2), (npix//2) - p[0]])

def larkin_transform(model, data):
    """
    Deconvolves the known PSF from the data using the Larkin transform
    This automatically pads and depads the array
    
    model: The knows PSF we want to deconvolve from the image
    data: The image we want deconvolved 
    
    The p value has been determined experimatally but should be further 
    tested to find the best value or the best method
    """
    # Pad 
    model_p, shape = pad_array(model)
    data_p, shape = pad_array(data)
    
    # Dive into Larkin-land
    M = np.fft.fft2(model_p) # M
    M_conj = np.conj(M) # M*
    D = np.fft.fft2(data_p) # D

    # Perform multiplications
    numer = np.multiply(D, M_conj) # D.M*
    denom = np.multiply(M, M_conj) # M.M*
    
    # Convolve with hamming function as a blurring method
    h = np.hamming(2)
    h2d = np.sqrt(np.outer(h,h))
    ham2d = np.zeros(model_p.shape)
    ham2d[:2, :2] = h2d
        
    # Convolve with hamming (apply blurring)
    numer_out = convolve(numer, ham2d)
    denom_out = convolve(denom, ham2d)
    
    # Prevent div by zero error 
    p = 1e-5
    sig = p*np.max(denom_out)
    denom_out += sig

    # Divide, inverse transform, take magnitude array
    div = np.divide(numer_out, denom_out)
    K = np.abs(np.fft.fftshift(np.fft.ifft2(div)))
    
    cen = shape[0]
    size = shape[0]//2
    K_out = K[cen-size:cen+size, cen-size:cen+size]

    return np.rot90(K_out, k=2)