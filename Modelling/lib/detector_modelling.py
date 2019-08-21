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
    
    BUG: some values are returned as negative (can fix with np.abs?)
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