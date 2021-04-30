import os
import sys
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
from scipy import ndimage
from scipy import fftpack

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
    if len(inputs) == 12:
        aperture, m2_obsc, npix, foc_length, pix_size, transform_size, sim_name, n, wl, r, phi, star = inputs
        pupil = np.load("{}/pupils/{}.npy".format(sim_name, wl))
        wavefront_error = None
    elif len(inputs) == 13:
        aperture, m2_obsc, npix, foc_length, pix_size, transform_size, sim_name, n, wl, r, phi, star,wfe_terms = inputs
        pupil = np.load("{}/pupils/{}.npy".format(sim_name, wl))
        wavefront_error = add_zernikes(wfe_terms,pupil.shape)
    else:
        raise Exception('Wrong number of params in model_image')
        
    PSF = model_PSF(pupil, aperture, m2_obsc, npix, wl*1e-9, foc_length, pix_size, transform_size, r, phi,
                    wavefront_error=wavefront_error)
    np.save("{}/PSFs/{}_{}_{}.npy".format(sim_name, n, wl, star), PSF)
    
def model_PSF(pupil, aperture, m2_obsc, npix, wl, focal_length, pix_size, transform_size, r, phi,wavefront_error=None):
    """
    Models the PSF of the optical system 
    wavefront_error is a 2D map of the wavefront error in the pupil (in nm). It must be the same shape as pupil.
    """
    
    # Convert
    r_rad = r * pix_size/focal_length
    
    # Apply the phase gradient to the pupil
    pupil_new = apply_phase_gradient(pupil, aperture, wl, r_rad, phi)
    
    # Apply wavefront error if we've added it
    if not type(wavefront_error) == type(None):
        # Check it's the right shape
        assert pupil_new.shape == wavefront_error.shape
        
        pupil_new *= np.exp(1j*wavefront_error/wl)
    
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

def make_zernike(array_shape,m,n,amplitude=1):
    """
    Make a zernike function on a circular aperture.
    Assumes the aperture fills the array.
    m,n define the polynomial and must be integers.
    
    Coordinates are defined relative to the corner of the 4 middle pixels, 
      not the centre of pixel (array_shape//2,array_shape//2).
    """

    radii_range = np.arange(array_shape[0])-array_shape[0]/2
    x,y = np.meshgrid(radii_range, radii_range)
    
    radius = np.hypot(x,y) / (array_shape[0]/2)
    theta = np.arctan2(x,y)
    def R_func(n,m,radius):
        R = np.zeros(radius.shape)
        for k in range(0,(n-m)//2+1):
            fact = (-1)**k * np.math.factorial(n-k)
            fact /= np.math.factorial(k) * np.math.factorial((n+m)/2 - k)
            fact /= np.math.factorial((n-m)/2 - k)
            R += fact * radius**(n-2*k)
        return R

    # Make sure the m,n values make sense.
    assert n >= m
    assert ((n-m) % 2) == 0
    
    if m >= 0:
        R = R_func(n,m,radius)
        z = R*np.cos(m*theta)
    else:
        R = R_func(n,-m,radius)
        z = R*np.sin(m*theta)
    # zero out the region outside the aperture
    # todo: include proper aperture size, not just the radius of the array
    z[radius>1] = 0
    return z*amplitude

def add_zernikes(wfe_terms,array_shape):
    """
    Takes a dictionary of wavefront error terms and produces 
    a single wavefront error map
    """
    wfe_map = np.zeros(array_shape)
    for wfe_coeffs in wfe_terms.keys():
        wfe_amplitude = wfe_terms[wfe_coeffs]  
        m,n = wfe_coeffs
        
        wfe_map += make_zernike(array_shape,m,n,amplitude=wfe_amplitude) # symmetric term
    return wfe_map

def fft_rotate(in_frame, alpha, pad=4, return_full = False):
    """
    3 FFT shear based rotation, following Larkin et al 1997
    
    Copied from the GRAPHIC exoplanet direct imaging pipeline

    in_frame: the numpy array which has to be rotated
    alpha: the rotation alpha in degrees
    pad: the padding factor
    return_full: If True, return the padded array

    One side effect of this program is that the image gains two columns and two rows.
    This is necessary to avoid problems with the different choice of centre between
    GRAPHIC and numpy. Numpy rotates around the boundary between 4 pixels, whereas this
    program rotates around the centre of a pixel.

    Return the rotated array
    """

    #################################################
    # Check alpha validity and correcting if needed
    #################################################
    alpha=1.*alpha-360*np.floor(alpha/360)

    # We need to add some extra rows since np.rot90 has a different definition of the centre
    temp = np.zeros((in_frame.shape[0]+3,in_frame.shape[1]+3))+np.nan
    temp[1:in_frame.shape[0]+1,1:in_frame.shape[1]+1]=in_frame
    in_frame = temp

    # FFT rotation only work in the -45:+45 range
    if alpha > 45 and alpha <= 135:
        in_frame=np.rot90(in_frame, k=1)
        alpha_rad=-np.deg2rad(alpha-90)
    elif alpha > 135 and alpha <= 225:
        in_frame=np.rot90(in_frame, k=2)
        alpha_rad=-np.deg2rad(alpha-180)
    elif alpha > 225 and alpha <= 315:
        in_frame=np.rot90(in_frame, k=3)
        alpha_rad=-np.deg2rad(alpha-270)
    else:
        alpha_rad=-np.deg2rad(alpha)

         # Remove one extra row
    in_frame = in_frame[:-1,:-1]

    ###################################
    # Preparing the frame for rotation
    ###################################

    # Calculate the position that the input array will be in the padded array to simplify
    #  some lines of code later
    px1=np.int(((pad-1)/2.)*in_frame.shape[0])
    px2=np.int(((pad+1)/2.)*in_frame.shape[0])
    py1=np.int(((pad-1)/2.)*in_frame.shape[1])
    py2=np.int(((pad+1)/2.)*in_frame.shape[1])

    # Make the padded array
    pad_frame=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
    pad_mask=np.ones((pad_frame.shape), dtype=bool)
    pad_frame[px1:px2,py1:py2]=in_frame
    pad_mask[px1:px2,py1:py2]=np.where(np.isnan(in_frame),True,False)

    # Rotate the mask, to know what part is actually the image
    pad_mask=ndimage.interpolation.rotate(pad_mask, np.rad2deg(-alpha_rad),
          reshape=False, order=0, mode='constant', cval=True, prefilter=False)

    # Replace part outside the image which are NaN by 0, and go into Fourier space.
    pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)


    ###############################
    # Rotation in Fourier space
    ###############################
    a=np.tan(alpha_rad/2.)
    b=-np.sin(alpha_rad)

    M=-2j*np.pi*np.ones(pad_frame.shape)
    N=fftpack.fftfreq(pad_frame.shape[0])

    X=np.arange(-pad_frame.shape[0]/2.,pad_frame.shape[0]/2.)#/pad_frame.shape[0]

    pad_x=fftpack.ifft((fftpack.fft(pad_frame, axis=0,overwrite_x=True).T*
        np.exp(a*((M*N).T*X).T)).T, axis=0,overwrite_x=True)
    pad_xy=fftpack.ifft(fftpack.fft(pad_x,axis=1,overwrite_x=True)*
        np.exp(b*(M*X).T*N), axis=1,overwrite_x=True)
    pad_xyx=fftpack.ifft((fftpack.fft(pad_xy, axis=0,overwrite_x=True).T*
        np.exp(a*((M*N).T*X).T)).T,axis=0,overwrite_x=True)

    # Go back to real space
    # Put back to NaN pixels outside the image.

    pad_xyx[pad_mask]=np.NaN


    if return_full:
        return np.abs(pad_xyx).copy()
    else:
        return np.abs(pad_xyx[px1:px2,py1:py2]).copy()
    
    
def add_photon_noise(image,peak_photons=None):
    """ Add photon noise to an image.
    peak_photons is the number of counts in the maximum pixel value of the image.
    """
    if peak_photons != None:
        image *= peak_photons/image.max()
        image = np.random.poisson(lam=image,size=image.shape)
    return image
