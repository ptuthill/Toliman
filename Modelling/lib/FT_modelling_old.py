import numpy as np
from lib.conversions import *
from scipy.ndimage import zoom

def pupil_phase_mask(pupil, aperture, wavelength, azimuthal_offset, angular_offset, shift=False):
    """
    Calculates the change in phase across the pupil induced by an off centre star
    pupil: complex array representing the phase pupil
    aperture: Aperture of the telescope (m)
    wavelength: Wavelength to be modelled (m)
    azimuthal_offset: Angle that is formed between the star being modelled and the normal of the telesope aperture (arcseconds)
    angular_offset: Angle around the circle formed by azimulath offset, 0 = offset in +x, pi/2 = offest in +y etc (radians)
    """
    # Calcuate the needed  values
    gridsize = pupil.shape[0]
    phi = arcsec_to_rad(azimuthal_offset)
    OPD = aperture*np.tan(phi)
    cycles = OPD/wavelength
    period = gridsize/cycles

    # Calculate the phase change over the pupil
    Xs = np.linspace(-gridsize//2,(gridsize//2)-1, num=gridsize)
    X, Y = np.meshgrid(Xs, Xs)
    r = np.hypot(X, Y)
    theta = np.arctan2(X, Y) - angular_offset 
    y_new = r * np.sin(theta)
    
    if shift:
        phase_shift = period/2
    else:
        phase_shift = period
    
    y_new += phase_shift
    eta = y_new * 2*np.pi/period
    phase_array = eta
    
    return phase_array

def pupil_phase_driver(pupil, aperture, wavelength, azimuthal_offset, angular_offset):
    """
    Calculates the change in phase across the pupil induced by an off centre star
    pupil: complex array representing the phase pupil
    aperture: Aperture of the telescope (m)
    wavelength: Wavelength to be modelled (m)
    azimuthal_offset: Angle that is formed between the star being modelled and the normal of the telesope aperture (arcseconds)
    angular_offset: Angle around the circle formed by azimulath offset, 0 = offset in +x, pi/2 = offest in +y etc (radians)
    """
    if azimuthal_offset == 0:
        return pupil
    
    phase = pupil_phase_mask(pupil, aperture, wavelength, azimuthal_offset, angular_offset, shift=False)
    anti_phase = pupil_phase_mask(pupil, aperture, wavelength, azimuthal_offset, angular_offset, shift=True)
    
    mag_array = np.abs(pupil)
    mask = np.angle(pupil) > 0
    inv_mask = -(mask - 1)
    
    selection1 = np.multiply(phase, inv_mask)
    selection2 = np.multiply(anti_phase, mask)
    selection = selection1 + selection2
    
    aperture_phase = mag_array * selection
    new_pupil = mag_array * np.exp(1j*aperture_phase)
    
    return new_pupil

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
    
    print(array_size)
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

def model_FT_scaled(mask, aperture, chip_dim, wl, foc_length, pix_size, FT_size):
    """
    Inputs:
        mask: Phase mask complex array 
        aperture: Telescope aperture diameter (m)
        chip dim: Units of pixels (int)
        wl: Wavelength (m)
        foc_lenght: Distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
        FT_size: Size of the desired FT array
    """
    
    # Decompose mask into Re and Im for scaling
    Re = mask.real
    mask_size = mask.shape[0]
    
    # Caclaute hoew big the pupil array needs to be to get our desired Fourier Transformed array size
    plate_scale = pix_size / foc_length    # Radians per pixel
    spatial_freq = wl/aperture
    scale_factor = spatial_freq/plate_scale
    size_out = int(FT_size/scale_factor)
    
    # Scale the real component of the pupil to this size
    size_in = mask_size
    ratio = size_out/size_in    
    Re_scaled = zoom(Re, ratio)
    
    # Zero the components outside of the aperture to zero
    c = size_out//2
    s = np.linspace(-c, c-1, num=size_out)
    X,Y = np.meshgrid(s, s)
    R = np.hypot(X,Y)
    Re_scaled[R > c-1] = 0 
    # Note the -1 here is to account for rounding errors intoruced by interger divisions
    
    # Create the new mask from a real and imaginary component
    mask_scaled = Re_scaled + np.zeros([size_out, size_out]) * 1j
    
    # Perform the FT
    array_out = np.zeros((FT_size,FT_size),dtype=complex)
    array_out[0:size_out, 0:size_out] = mask_scaled
    im = np.fft.fftshift(np.abs(np.fft.fft2(array_out))**2)
        
    # Take the part that falls on the chip
    # Note: This could intorduce some form of float error (ie returns a 9x9 instead of 10x10 array)
    start = (FT_size-chip_dim)//2
    end = FT_size - (FT_size-chip_dim)//2
    im_out = im[start:end, start:end]
    
    return im_out

def on_axis_FT(mask, aperture, chip_dim, wl, foc_length, pix_size, FT_size):
    """
    Inputs:
        mask: Phase mask complex array 
        aperture: Telescope aperture diameter (m)
        chip dim: Units of pixels (int)
        wl: Wavelength (m)
        foc_lenght: Distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
        FT_size: Size of the desired FT array
    """
    
    # Decompose mask into Re and Im for scaling
    Re = mask.real
    mask_size = mask.shape[0]
    
    # Caclaute hoew big the pupil array needs to be to get our desired Fourier Transformed array size
    plate_scale = pix_size / foc_length    # Radians per pixel
    spatial_freq = wl/aperture
    scale_factor = spatial_freq/plate_scale
    size_out = int(FT_size/scale_factor)
    
    # Scale the real component of the pupil to this size
    size_in = mask_size
    ratio = size_out/size_in    
    Re_scaled = zoom(Re, ratio)
    
    # Zero the components outside of the aperture to zero
    c = size_out//2
    s = np.linspace(-c, c-1, num=size_out)
    X,Y = np.meshgrid(s, s)
    R = np.hypot(X,Y)
    Re_scaled[R > c-1] = 0 
    # Note the -1 here is to account for rounding errors intoruced by interger divisions
    
    # Create the new mask from a real and imaginary component
    mask_scaled = Re_scaled + np.zeros([size_out, size_out]) * 1j
    
    # Perform the FT
    array_out = np.zeros((FT_size,FT_size),dtype=complex)
    array_out[0:size_out, 0:size_out] = mask_scaled
    im = np.fft.fftshift(np.abs(np.fft.fft2(array_out))**2)
        
    # Take the part that falls on the chip
    # Note: This could intorduce some form of float error (ie returns a 9x9 instead of 10x10 array)
    start = (FT_size-chip_dim)//2
    end = FT_size - (FT_size-chip_dim)//2
    im_out = im[start:end, start:end]
    
    return im_out

def off_axis_FT(mask, aperture, chip_dim, wl, foc_length, pix_size, FT_size, binary_separation, field_angle):
    """
    Inputs:
        mask: Phase mask complex array 
        aperture: Telescope aperture diameter (m)
        chip dim: Units of pixels (int)
        wl: Wavelength (m)
        foc_lenght: Distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
        FT_size: Size of the desired FT array
    """
    
    # Decompose mask into Re and Im for scaling
    Re = mask.real
    mask_size = mask.shape[0]
    
    # Caclaute hoew big the pupil array needs to be to get our desired Fourier Transformed array size
    plate_scale = pix_size / foc_length    # Radians per pixel
    spatial_freq = wl/aperture
    scale_factor = spatial_freq/plate_scale
    size_out = int(FT_size/scale_factor)
    
    # Scale the real component of the pupil to this size
    size_in = mask_size
    ratio = size_out/size_in    
    Re_scaled = zoom(Re, ratio)
    
    # Zero the components outside of the aperture to zero
    c = size_out//2
    s = np.linspace(-c, c-1, num=size_out)
    X,Y = np.meshgrid(s, s)
    R = np.hypot(X,Y)
    Re_scaled[R > c-1] = 0 
    # Note the -1 here is to account for rounding errors intoruced by interger divisions
    
    # Create the new mask from a real and imaginary component
    mask_scaled = Re_scaled + np.zeros([size_out, size_out]) * 1j
    
    # Apply the phase gradient to the pupil
    off_axis_mask = pupil_phase_driver(mask_scaled, aperture, wl, binary_separation, field_angle)
    
    # Perform the FT
    array_out = np.zeros((FT_size,FT_size),dtype=complex)
    array_out[0:size_out, 0:size_out] = off_axis_mask
    im = np.fft.fftshift(np.abs(np.fft.fft2(array_out))**2)
        
    # Take the part that falls on the chip
    # Note: This could intorduce some form of float error (ie returns a 9x9 instead of 10x10 array)
    start = (FT_size-chip_dim)//2
    end = FT_size - (FT_size-chip_dim)//2
    im_out = im[start:end, start:end]
    
    return im_out