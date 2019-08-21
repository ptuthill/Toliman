"""
A script for all the functions that are called from various other scripts
None of these functions should be called from either driver.py or the main notebook
"""
from math import tan
import numpy as np

def fringes_to_pixels(num_fringes, ap, fl, wl, dp):
    """
    converts a fringe radius to a pixel radius
    fl: focal length
    wl: wavelength
    ap: aperture
    dp: detector pitch
    all units in meters
    """
    return fl*tan(num_fringes*wl/ap)/dp

def pad_array(array, size):
    """
    Inputs:
        array, 2D complex array: array representing the modified wavefront at the detector
        size, int: the scaling or padding factor, determined by trial and error for given inputs
    Outputs:
        padded, 2D complex array: the padded array
    """
    # Get realative sizes
    array_size = array.shape[0]
    pad = (size - array_size)//2
    
    # Pad WF array
    padded = np.zeros([size, size], dtype=complex)
    padded[pad:array_size+pad, pad:array_size+pad] = array
    return padded


def model_FT(mask, mask_size, chip_dim, wavels, foc_length, pix_size, power=True):
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
    
    if power:
        im_out = np.zeros((chip_dim,chip_dim))
    else:
        im_out = np.zeros((chip_dim,chip_dim), dtype=complex)
    
    for wavel in wavels:
        spatial_freq = wavel/mask_size
        array_size = int(grid_size*spatial_freq/plate_scale)
        complex_array = np.array(np.zeros((array_size,array_size)),dtype=complex)
        complex_array[0:grid_size,0:grid_size] = mask
        
        if power:
            im = np.fft.fftshift(np.abs(np.fft.fft2(complex_array))**2)
        else:
            im = np.fft.fftshift(np.fft.fft2(complex_array))
            
        # Vector or matrix operation
        # Scipy or numpy regridding
        for y in range(chip_dim):
            for x in range(chip_dim):
                # Split line below and multiply line below by some scale/ratio to normalise
                # Pass in all wavelengths across all bands together and sum then normalise to 1
                im_out[y][x] += im[int((array_size-chip_dim)/2) + y][int((array_size-chip_dim)/2) + x]

    return im_out


#############################
# Redundant functions below #
#############################

# Redundant
def generate_array_from_fits(fits_file, offset=0):
    """
    Takes in the fits file and returns a complex array of the pupil
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

# Redundant
def gaussian(x, y, p, mu, sig):
    """
    Returns the gaussian value for an x and y value based on the inputs 
    Note: efficiency can be increased by pre computing values unchanged by x and y inputs (ie sig and mu calcs)
    """
    a = 1 / (sig * np.sqrt(2 * np.pi)) 
    b = ((x-mu)**2)/(2*sig**2)
    c = ((y-mu)**2)/(2*sig**2)
    g_out = a*np.exp(-(b+c)**p)
    return g_out

# Redundant
def create_overlay(array_size, sigma, p=10, mu=0):
    """
    Creates a gaussian overlay for an given array
    assumes a square input
    """   
    array_out = np.zeros([array_size, array_size])
    c = array_size//2
    
    for i in range(array_size):
        for j in range(array_size):
            x = i-c 
            y = j-c
            array_out[i][j] = gaussian(x, y, p, mu, sigma)
            
    return array_out
            
# Redundant
def pad_WF_to_pupil(pupil_object, WF):
    """
    Pads the WF array to match the size of the pupil for future fourrier transforming
    Assumes square inputs
    NOTE: REDUNDANT, REPLACED BY pad_array()
    """    
    # Get realative sizes
    pupil_size = pupil_object["pupil_inputs"]["array_size"]
    WF_size = WF.shape[0]
    pad = (pupil_size - WF_size)//2
    
    # Pad WF array
    WF_pad = np.zeros([pupil_size, pupil_size], dtype=complex)
    WF_pad[pad:WF_size+pad, pad:WF_size+pad] = WF
    return WF_pad
        
# May be usefull later
def create_symmetry(A, f, binarise=False):
    from math import sin, cos, hypot, atan2
    
    # A is array to create symmetry with
    # f is the number of rotations

    rows = A.shape[0]
    cols = A.shape[1]
    c = rows//2
    B = np.zeros((rows, cols)) 
    alpha = 2*np.pi/f 
    
    max_val = np.max(A)
    min_val = np.min(A)
    threshold = max_val/2

    # Iterate over each entry in the array
    for i in range(rows):
        for j in range(cols):
            x = i - c
            y = j - c
            r = hypot(x, y)
            theta = atan2(y, x)

            # Only perform operation on pixels within the radius
            if r <= c:
                for k in range(f):
                    theta_raw = theta - k*alpha
                    theta_new = theta_raw if theta_raw >= -np.pi else 2*np.pi + theta_raw
                    
                    x_new = round(r*cos(theta_new))
                    y_new = round(r*sin(theta_new))

                    k = x_new - c
                    l = y_new - c

                    B[i][j] += A[k][l]/f
                    
            else:
                continue
    return B
        


