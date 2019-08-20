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
    Inputs:
        mask: Phase mask complex array 
        mask_size: Size of the phase mask, ie the aperture diameter (m)
        chip dim: Units of num pixels
        wavels: Array of wavelengths (m)
        foc_lenght: Focal length of lens/distance to focal plane of telescope (m)
        pix_size: Detector pitch, size of pixel unit cell (m)
        power: True returns |im|^2, False returns eletric field complex array
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