import codecs
import numpy as np

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