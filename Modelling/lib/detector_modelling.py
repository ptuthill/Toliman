import numpy as np
from scipy.ndimage import shift, zoom
from lib.conversions import *

def CCD_output(photons, QE, read_noise, dark_current, fps, gain, full_well):
    """
    Models the convertion of a psf incident to the detector into an image
    QE: Quantum efficiency [0, 1]
    read_noise: electrons per pixel
    dark_current: electrons / (second * pixel)
    fps: frames per second
    gain: electrons / ADU
    full_well: electrons/pixel
    
    primarily put together with info from: http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/
    
    BUG: some values are returned as negative (fixed with np.abs?)
    """
    # Poission noise of photons 
    photons = np.random.poisson(np.abs(photons)) 
#     photons = np.random.poisson(photons) 
    
    # Convert photons to electrons 
    electrons = np.round(photons * QE) 
    
    # Calcuate total electron noise
    dark_noise = read_noise + (dark_current/fps) 
    
    # Model noise with gaussian distribution
    noise = np.round(np.random.normal(scale=dark_noise, size=electrons.shape)) 
    
    # Add noise CCD noise to image and get ADU output
    # Should this be sensitivity rather than gain?? 
    image = (electrons + noise) * gain 
    
    # Model pixel saturation
    image[image > full_well] = full_well 
    
    return image

def interpolate_to_detector(PSFs, azimuthal_offset, pixel_size, detector_pitch, focal_length, num_positions):
    """
    Interpolates and downsamples the high res PSFs produced by zemax
    Inputs:
        PSFs, list:
            - [0] on-axis PSF
            - [1] off-axis PSF
        azimuthal_offset, arcsec: Angular offset from the central FoV
        pixel_size, m: Size of the pixels used to form the PSFs in Zemax
        detector_pitch, m: Size of the pixels on the detector
        focal_length, m: Focal lenght of the telescope
        num_positions, int: The number of angular position about the centre to simulate 
        
    Outputs: 
        detector_psfs, list: List of 2D array representing the PSFs interpolated onto the detector
    """
    # Calcualte needed values
    pixel_seperation = arcsec_to_pixel(azimuthal_offset, pixel_size, focal_length)
    ratio = pixel_size/detector_pitch
    
    # Calculate the x-y pixel shift coordinates
    thetas = np.linspace(-np.pi, np.pi, num=num_positions)
    Xs = pixel_seperation*np.cos(thetas)
    Ys = pixel_seperation*np.sin(thetas)
    
    # Interpolate PSFs
    psfs_oversampled = [shift(PSFs[0], [0, 0]) + shift(PSFs[1], [Xs[i], Ys[i]]) for i in range(num_positions)]

    # Downsample PSFs
    
    detector_psfs = [zoom(psf, ratio) for psf in psfs_oversampled]
    
    return detector_psfs

def interpolate_bayer(image):
    """
    Takes in a detector image and does a full bayer interpolation to simulate the output of a bayer array
    """
    red = get_red(image)
    green = get_green(image)
    blue = get_blue(image)
    return [red, green, blue]

def get_red(image):
    """
    Returns the red channel from the RGGB bayer pattern
    Assumes and outputs a square image
    """
    chip_dim = image.shape[1]
    bayer_red = np.remainder((np.arange(chip_dim)),2)
    bayer_red = np.outer(bayer_red,bayer_red)
    im_red = np.zeros((chip_dim//2,chip_dim//2))
    im_red = (image[bayer_red.nonzero()]).reshape(chip_dim//2,chip_dim//2)
    return im_red

def get_green(image):
    """
    Returns the green channel from the RGGB bayer pattern
    Assumes and outputs a square image
    Returned image is rotated by 45 degrees and has null output for the corners
    """
    chip_dim = image.shape[1]
    bayer_green = np.remainder((np.arange(chip_dim**2))+np.arange(chip_dim**2)/chip_dim,2)
    bayer_green = bayer_green.reshape(chip_dim,chip_dim)
    gnz = bayer_green.nonzero() 
    g_mapping = [chip_dim//2 -1 - (gnz[0]-gnz[1])//2,(gnz[0]+gnz[1])//2]
    im_green=np.zeros((chip_dim,chip_dim))
    im_green[g_mapping] = image[gnz]
    return im_green
    
def get_blue(image):
    """
    Returns the blue channel from the RGGB bayer pattern
    Assumes and outputs a square image
    """
    chip_dim = image.shape[1]
    bayer_blue = np.remainder((np.arange(chip_dim))+1,2)
    bayer_blue = np.outer(bayer_blue,bayer_blue)
    im_blue = np.zeros((chip_dim//2,chip_dim//2))
    im_blue = (image[bayer_blue.nonzero()]).reshape(chip_dim//2,chip_dim//2)
    return im_blue