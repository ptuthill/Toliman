import codecs
import numpy as np
import pysynphot as S

def zemax_to_array(path_to_file):
    with open(path_to_file,'rb') as f:
        contents = f.read()
        
    contents = contents.decode("utf-16").split("\n")
    data_raw = [line.strip() for line in contents]
    
    metadata = []
    data = []
    
    for line in data_raw[:20]:
        if line != '':
            metadata.append(line)
            
    for line in data_raw[21:-1]:
        line = line.split("\t  ")
        line_formatted = [float(l) for l in line if l != '']
        data.append(line_formatted)
        
    return np.asarray(data), metadata

def model_FT(mask, mask_size, chip_dim, wavels, foc_length, pix_size):
    """
    Redundant version, much slower use model_FT_monochromatic/broadband
    
    Inputs:
        mask: Phase mask complex array 
        mask_size: Size of the phase mask, ie the aperture diameter (m)
        chip dim: Units of num pixels
        wavels: Array of wavelengths (m)
        foc_lenght: Focal length of lens/distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
    Note: Assumes all wavelengths have equal intesity (add intesities later)
    """
    grid_size = mask.shape[1]
    plate_scale = pix_size / foc_length    # Radians per pixel
    
    im_out = np.zeros((chip_dim,chip_dim))

    for wavel in wavels:
        spatial_freq = wavel/mask_size
        array_size = int(grid_size*spatial_freq/plate_scale)
        complex_array = np.array(np.zeros((array_size,array_size)),dtype=complex)
        complex_array[0:grid_size,0:grid_size] = mask
        im = np.fft.fftshift(np.abs(np.fft.fft2(complex_array))**2)
            
        # Vector or matrix operation
        # Scipy or numpy regridding
        for y in range(chip_dim):
            for x in range(chip_dim):
                # Split line below and multiply line below by some scale/ratio to normalise
                # Pass in all wavelengths across all bands together and sum then normalise to 1
                im_out[y][x] += im[int((array_size-chip_dim)/2) + y][int((array_size-chip_dim)/2) + x]

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
    """
    grid_size = mask.shape[1]
    plate_scale = pix_size / foc_length    # Radians per pixel
    
    im_out = np.zeros((chip_dim,chip_dim))

    for wavel, weights in zip(wavels, weights):
        spatial_freq = wavel/mask_size
        array_size = int(grid_size*spatial_freq/plate_scale)
        complex_array = np.array(np.zeros((array_size,array_size)),dtype=complex)
        complex_array[0:grid_size,0:grid_size] = mask
        im = np.fft.fftshift(np.abs(np.fft.fft2(complex_array))**2)
        
        # This could intorduce some form of float error (returns a 9x9 instead of 10x10 array)
        start = (array_size-chip_dim)//2
        end = array_size - (array_size-chip_dim)//2
        im_out += weight * im[start:end, start:end]

    return im_out

def model_FT_monochromatic(mask, mask_size, chip_dim, wavel, foc_length, pix_size):
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
    complex_array = np.array(np.zeros((array_size,array_size)),dtype=complex)
    complex_array[0:grid_size, 0:grid_size] = mask
    im = np.fft.fftshift(np.abs(np.fft.fft2(complex_array))**2)
        
    # Note: This could intorduce some form of float error (ie returns a 9x9 instead of 10x10 array)
    start = (array_size-chip_dim)//2
    end = array_size - (array_size-chip_dim)//2
    im_out = im[start:end, start:end]

    return im_out

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
    photons = np.random.poisson(photons) 
    
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

def get_countrate(aperture, central_obscuration, nwavels, wl_range, star_dict):
    """
    Aperture (cm): Aperture of the telescope
    central_obscuration (cm): diameter of central obscuration of telescope
    nwaels: number of wavelengths to sample
    wl_range (Angstroms): [first wavelength, last wavelength] 
    star_dict: dictionary of the following structure
        {"mag": float, # vega mag
        "Teff": int,
        "Z": float,
        "log g": float}        
    Uses the phoenix models as defined from pysynphot docs
    """
    
    r = (aperture-central_obscuration)/2
    collecting_area = (np.pi * r**2)
    S.refs.setref(area=collecting_area) # Takes units of cm^2
    wavels = np.linspace(wl_range[0], wl_range[1] ,nwavels)
    throughput = np.ones(nwavels)
    bandpass = S.ArrayBandpass(wavels, throughput)
    
    
    star_obj = S.Icat('phoenix', star_dict["Teff"], star_dict["Z"], star_dict["log g"])
    spec_filt = star_obj.renorm(star_dict["mag"], 'vegamag', bandpass)
    obs = S.Observation(spec_filt, bandpass, binset=wavels)
    
    return obs.countrate()

def rad_to_asec(angle):
    return angle*3600*180/np.pi

def asec_to_rad(angle):
    return angle * np.pi/(180 * 3600)


def generate_array_from_fits(fits_file, offset=0):
    """
    Takes in the fits file and returns a complex array of the pupil
    
    Can be sped up by vectorisation
    """
    import math

    gridsize = fits_file.shape[0] - 2*offset
    c = gridsize//2
    pupil = np.zeros((gridsize-offset,gridsize-offset), dtype=complex)

    for i in range(gridsize):
        for j in range(gridsize):
            x = i - c
            y = j - c 
            r = math.hypot(x, y)
            if r >= (gridsize//2) + offset:
                pupil[i][j] = np.complex(0,0)
            else:
                pupil[i][j] = np.exp(1j*fits_file[i][j])
        
    return pupil

def pupil_phase_offset_old(pupil, aperture, wavelength, azimuthal_offset, angular_offset):
    """
    Redundant, faster vectorised version as pupil_phase_offset
    
    Calculates the change in phase across the pupil induced by an off centre star
    pupil: complex array representing the phase pupil
    aperture: Aperture of the telescope (m)
    wavelength: Wavelength to be modelled (m)
    azimuthal_offset: Angle that is formed between the star being modelled and the normal of the telesope aperture (arcseconds)
    angular_offset: Angle around the circle formed by azimulath offset, 0 = offset in +x, pi/2 = offest in +y etc (radians)
    """
    gridsize = pupil.shape[0]
    offset = asec_to_rad(azimuthal_offset)
    OPD = aperture*np.tan(offset)
    cycles = OPD/wavelength
    period = gridsize/cycles
    phase_array = np.zeros([gridsize, gridsize])
    
    for x in range(gridsize):
        for y in range(gridsize):

            # Shift the coordinates to get an off axis sine wave
            r = np.hypot(x, y)
            theta = np.arctan2(y, x) - angular_offset
            y_new = r*np.sin(theta)

            # Calcualte the sine wave input value in the new coordiante system
            eta = y_new * 2*np.pi/period

            # Try changing seocnd if statement to no remainder
            if eta%(2*np.pi) < np.pi/2 or eta%(2*np.pi) >= 3*np.pi/2:
                phase_array[x][y] = np.sin(eta)*np.pi
            else:
                phase_array[x][y] = -np.sin(eta)*np.pi
                
    pupil_out = np.zeros([gridsize, gridsize], dtype=complex)
    for i in range(gridsize):
        for j in range(gridsize):
            mag = np.abs(pupil[i][j])

            if mag == 0:
                pupil_out[i][j] = np.complex(0,0)

            else:
                angle = phase_array[i][j] - np.angle(pupil[i][j]) 
                pupil_out[i][j] = mag*np.exp(1j*angle)
                
    return pupil_out
    
    
def pupil_phase_offset(pupil, aperture, wavelength, azimuthal_offset, angular_offset):
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
    offset = asec_to_rad(azimuthal_offset)
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
    
    return pupil_out
    