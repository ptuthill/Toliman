import numpy as np
from lib.conversions import *
# from conversions import *

def pupil_phase_offset_test(pupil, aperture, wavelength, azimuthal_offset, angular_offset):
    """
    Calculates the change in phase across the pupil induced by an off centre star
    pupil: complex array representing the phase pupil
    aperture: Aperture of the telescope (m)
    wavelength: Wavelength to be modelled (m)
    azimuthal_offset: Angle that is formed between the star being modelled and the normal of the telesope aperture (arcseconds)
    angular_offset: Angle around the circle formed by azimulath offset, 0 = offset in +x, pi/2 = offest in +y etc (radians)
    """
    # Calcuate the needed values
    gridsize = pupil.shape[0]
    offset = arcsec_to_rad(azimuthal_offset)
    OPD = aperture*np.tan(offset)
    cycles = OPD/wavelength
    period = gridsize/cycles

    # Calculate the phase change over the pupil
    Xs = np.linspace(0,gridsize-1, num=gridsize)
    X, Y = np.meshgrid(Xs, Xs)
    r = np.hypot(X, Y)
    theta = np.arctan2(X, Y) - angular_offset
    y_new = r * np.sin(theta)
    eta = y_new * 2*np.pi/period
    eta_mod = eta%(2*np.pi)
    eta[eta_mod < np.pi/2] = -eta[eta_mod < np.pi/2]
    eta[eta_mod > 3*np.pi/2] = -eta[eta_mod > 3*np.pi/2]
    phase_array = np.sin(eta)*np.pi
    
    # Impose the phase array on to the pupil
    mag_array = np.abs(pupil)
    angle_array = np.angle(pupil)
    aperture_phase = mag_array*(phase_array - angle_array)
    pupil_out = mag_array * np.exp(1j*aperture_phase)
    
    return pupil_out, phase_array

def model_FT(mask, mask_size, chip_dim, wavel, foc_length, pix_size):
    """
    Inputs:
        mask: Phase mask complex array 
        mask_size: Size of the phase mask, ie the aperture diameter (m)
        chip dim: Units of num pixels
        wavel: Wavelength (m)
        foc_lenght: Focal length of lens/distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
    """
    grid_size = mask.shape[1]
    plate_scale = pix_size / foc_length    # Radians per pixel
    
    spatial_freq = wavel/mask_size
    array_size = int(grid_size*spatial_freq/plate_scale)
    complex_array = np.zeros((array_size,array_size), dtype=complex)
    complex_array[0:grid_size, 0:grid_size] = mask
    im = np.fft.fftshift(np.abs(np.fft.fft2(complex_array))**2)
        
    # Note: This could intorduce some form of float error (ie returns a 9x9 instead of 10x10 array)
    start = (array_size-chip_dim)//2
    end = array_size - (array_size-chip_dim)//2
    im_out = im[start:end, start:end]

    return im_out

def model_FT_broadband(mask, mask_size, chip_dim, wavels, weights, foc_length, pix_size):
    """
    Inputs:
        mask: Phase mask complex array 
        mask_size: Size of the phase mask, ie the aperture diameter (m)
        chip dim: Units of num pixels
        wavels: Array of wavelengths (m)
        weights: Relative weights of each wavelength
        foc_lenght: Focal length of lens/distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
        
    Note:
        Untested
    """
    im_out = np.zeros((chip_dim,chip_dim))
    
    for wavel, weight in zip(wavels, weights):
        im_out += weight * model_FT(mask, mask_size, chip_dim, wavel, foc_length, pix_size)

    return im_out
