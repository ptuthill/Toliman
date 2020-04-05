import os
import sys
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom

def make_pupil(inputs):
    """
    Pre computes and saves a series of scaled pupils to disk
    """
    size_out, wl, pupil_in, sim_name = inputs
    
    # Scale the pupil 
    pupil = scale_pupil(pupil_in, size_out)
    np.save("{}/pupils/{}.npy".format(sim_name, wl), pupil)
    
def model_image(inputs):
    """
    Drives the model_PSF function.
    Saves outputs to file to combine later
    """
    
    aperture, m2_obsc, npix, foc_length, pix_size, transform_size, sim_name, n, wl, r, phi, star = inputs
    pupil = np.load("{}/pupils/{}.npy".format(sim_name, wl))
    PSF = model_PSF(pupil, aperture, m2_obsc, npix, wl*1e-9, foc_length, pix_size, transform_size, r, phi)
    np.save("{}/PSFs/{}_{}_{}.npy".format(sim_name, n, wl, star), PSF)
    
def model_PSF(pupil, aperture, m2_obsc, npix, wl, focal_length, pix_size, transform_size, r, phi):
    """
    Models the PSF of the optical system 
    """
    
    # Convert
    r_rad = r * pix_size/focal_length
    
    # Apply the phase gradient to the pupil
    pupil_new = apply_phase_gradient(pupil, aperture, wl, r_rad, phi)
    
    # Create xy meshgrid
    radius = aperture/2
    radii_range = np.linspace(-radius, radius, num=pupil_new.shape[0], endpoint=False)
    X,Y = np.meshgrid(radii_range, radii_range)
    R = np.hypot(X,Y)
    
    # Zero the power to non aperture regions
    mag_array = np.ones(pupil_new.real.shape)
    mag_array[R > radius] = 0 
    mag_array[R < m2_obsc/2] = 0
    phase_array = np.angle(pupil_new)
    pupil_out = mag_array * np.exp(1j*phase_array)
        
    # Perform the FT
    transform_array = np.zeros((transform_size,transform_size),dtype=complex)
    transform_array[0:pupil_new.shape[0], 0:pupil_new.shape[1]] = pupil_out
    im = np.fft.fftshift(np.abs(np.fft.fft2(transform_array))**2)

    # Take the part that falls on the chip
    c = im.shape[0]//2
    s = npix//2
    im_out = im[c-s:c+s:, c-s:c+s]
    
    # Normalise to a sum of 1
    PSF_out = im_out/np.sum(im_out)    
    return PSF_out

def arcsec_to_pixel(azimuthal_offset, pixel_size, focal_length):
    """
    Converts an 'azimuthal offset' in arcseconds into a pixel value 
    Inputs:
        azimuthal_offset, arcsec: Angular offset from centre point of FoV
        pixel_size, m: Physical size of pixels in array
        focal_length, m: Focal length of the telescope
    """
    return azimuthal_offset / rad_to_arcsec(pixel_size / focal_length)

def rad_to_arcsec(angle):
    return angle * 3600 * 180 / np.pi

def pupil_from_fits(file_name):
    """
    Takes in the fits file and returns a complex array of the pupil
    """
    # Create a fits object from astropy
    fits_file = fits.open(file_name)[0].data
    array = np.array(fits_file)

    # Calculate needed values
    gridsize = array.shape[0]
    c = gridsize//2
    
    # Create value arrays
    Xs = np.linspace(-c, c-1, num=gridsize)
    X, Y = np.meshgrid(Xs, Xs)
    r = np.hypot(X, Y)
    
    # Create pupil
    pupil = np.exp(1j*array)
    
    # Zero outer regions
    pupil[r >= (gridsize//2)] = np.complex(0,0)
        
    return pupil

def scale_pupil(pupil, output_size):
    """
    Takes in the complex pupil array and returns a scaled version
    
    BUGS: 
    This does not work correctly on pupils with the side lobe gratings
    
    Inputs:
        pupil: 2D Complex np array of the phase pupil
        output_size: Size of the pupil array to be returned
        
    Returns:
        pupil_scaled: 2D Complex np array of the pupil in the dimension of output_size X output_size
    """
    # Scale the real component of the pupil to this size
    size_in = pupil.shape[0]
    ratio = output_size/size_in    
    Re_scaled = zoom(pupil.real, ratio)
    
    # Create the new pupil from a real and imaginary component
    pupil_scaled = Re_scaled + np.zeros([output_size, output_size]) * 1j
    
    # Create the masks to fix the values of the pupil
    norm_phase = np.logical_and(np.abs(pupil_scaled) != 0, np.angle(pupil_scaled) == 0)
    anti_phase = np.logical_and(np.abs(pupil_scaled) != 0, 
                                np.logical_or(np.angle(pupil_scaled) == np.pi, np.angle(pupil_scaled) == -np.pi))
    c = output_size//2
    s = np.linspace(-c, c, num=output_size, endpoint=False)
    X,Y = np.meshgrid(s, s)
    R = np.hypot(X,Y)
    
    # Assign the correct values to the new array
    pupil_scaled[norm_phase] = np.complex(1, 0)
    pupil_scaled[anti_phase] = np.complex(-1, 0)
    pupil_scaled[R >= c-1] = np.complex(0, 0)
    
    return pupil_scaled

def apply_phase_gradient(pupil, aperture, wavelength, r, phi):
    """
    Applies the change in phase across the pupil induced by an off centre star to the input binarised phase pupil
    
    Inputs:
        pupil: complex array representing the phase pupil
        aperture: Aperture of the telescope (m)
        wavelength: Wavelength to be modelled (m)
        r: radial offset from normal of telescope aperture in polar coordaintes (radians)
        phi: angular offset from the positive x plane (radians)
        
    Returns:
        Complex numpy array with the phase gradient for off-axis stars applied
    """
    if r == 0:
        return pupil
    
    phase = calculate_phase_gradient(pupil, aperture, wavelength, r, phi, shift=False)
    anti_phase = calculate_phase_gradient(pupil, aperture, wavelength, r, phi, shift=True)
    
    mask = np.angle(pupil) > 0
    inv_mask = -(mask - 1)
    
    selection1 = np.multiply(phase, inv_mask)
    selection2 = np.multiply(anti_phase, mask)
    aperture_phase = selection1 + selection2
    
    pupil_phase = np.exp(1j*aperture_phase)
    
    return pupil_phase

def calculate_phase_gradient(pupil, aperture, wavelength, r, phi, shift=False):
    """
    Calculates the change in phase across the pupil induced by an off centre star
    
    Inputs:
        pupil: complex array representing the phase pupil
        aperture: Aperture of the telescope (m)
        wavelength: Wavelength to be modelled (m)
        r: radial offset from normal of telescope aperture in polar coordaintes (radians)
        phi: angular offset from the positive x plane (radians)
        
    Returns: numpy array of the phase change of the incident light across the pupil apertuere 
    """
    # Create an xy coordinate grid
    gridsize = pupil.shape[0]        
    Xs = np.linspace(-gridsize//2,(gridsize//2), num=gridsize, endpoint=False)
    X, Y = np.meshgrid(Xs, Xs)
    
    # Convert to polar coords
    R = np.hypot(X, Y)
    theta_array = np.arctan2(X, Y) 
    
    # Rotate the array by the phi offset
    theta_array_shifted = theta_array - phi 
    
    # Convert back to cartesian coordiantes for typical use
    y_new = R * np.sin(theta_array_shifted)

    # Calculate the scaling values from the wavelength and offset
    OPD = aperture*np.tan(r)
    cycles = OPD/wavelength
    period = gridsize/cycles
    
    # Phase shift for reccessed regions of pupil
    if shift:
        phase_shift = period/2
    else:
        # Note this may be redundant - it was introduced to curb introduced distortions which may have been from an earier bug
        phase_shift = period
        
    # Apply phase shift to array
    y_new += phase_shift
    
    # Scale phase change based on calculated scaling values
    phase_array = y_new * 2*np.pi/period
    
    return phase_array