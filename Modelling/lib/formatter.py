import numpy as np
import imageio
from astropy.io import fits
from scipy.ndimage import zoom


def pupil_from_fits(file_name, offset=0, output_size=0):
    """
    Takes in the fits file and returns a complex array of the pupil
    """
    # Create a fits object from astropy
    fits_file = fits.open(file_name)[0].data
    array = np.array(fits_file)
    
    if output_size != 0:
        ratio = output_size/array.shape[0]
        scaled_array = zoom(array, ratio)

        # Some values seem to get changed in the process, this is an ad-hoc fix
        scaled_array[scaled_array >= np.pi] = np.pi
        scaled_array[scaled_array < 0] = 0
        
        array = scaled_array

    # Calculate needed values
    gridsize = array.shape[0] - 2*offset
    c = gridsize//2
    
    # Create value arrays
    Xs = np.linspace(-c, c-1, num=gridsize)
    X, Y = np.meshgrid(Xs, Xs)
    r = np.hypot(X, Y)
    
    # Create pupil
    pupil = np.exp(1j*array)
    
    # Zero outer regions
    pupil[r >= (gridsize//2) + offset] = np.complex(0,0)
        
    return pupil

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

def create_sag_file(file_name, pupil, aperture, unit, target_wl):
    """
    Creates a sag file formatted for Zemax
    file_name: Name for the file
    Pupil: complex array
    unit: unit of measurement (mm, cm, m, in)
    aperture: telescope aperture in units of 'unit'
    target_wl: ideal wavelength in units of 'unit'
    """
    unit_dict = {"mm": 0, "cm": 1, "in": 2, "m": 3}
    phase_range = 2*np.pi
    
    nx = pupil.shape[0]
    ny = pupil.shape[1]
    delx = aperture/nx
    dely = aperture/ny
    unitflag  = unit_dict[unit]
    xdec = 0
    ydec = 0
    
    with open("{}.DAT".format(file_name), 'w') as f:
        f.write("{} {} {} {} {} {} {}\n".format(nx, ny, delx, dely, unitflag, xdec, ydec))
        
        for i in range(nx):
            for j in range(ny):
                sag_ratio = np.angle(pupil[i][j])/phase_range
                sag_val = sag_ratio*target_wl/2
                if sag_val < 1e-12:
                    sag_val = 0
                    
                f.write("{} {} {} {} {}\n".format(float(sag_val), 0, 0, 0, 0))
                
def create_phase_file(file_name, pupil, aperture, unit):
    """
    Creates a sag file formatted for Zemax
    file_name: Name for the file
    Pupil: complex array
    unit: unit of measurement (mm, cm, m, in)
    aperture: telescope aperture in units of 'unit'
    """
    unit_dict = {"mm": 0, "cm": 1, "in": 2, "m": 3}
    
    nx = pupil.shape[0]
    ny = pupil.shape[1]
    delx = aperture/nx
    dely = aperture/ny
    unitflag  = unit_dict[unit]
    xdec = 0
    ydec = 0
    
    with open("{}.DAT".format(file_name), 'w') as f:
        f.write("{} {} {} {} {} {} {}\n".format(nx, ny, delx, dely, unitflag, xdec, ydec))
        
        for i in range(nx):
            for j in range(ny):
                phase_val = np.angle(pupil[i][j])
                if phase_val < 1e-12:
                    phase_val = 0
                f.write("{} {} {} {} {}\n".format(phase_val, 0, 0, 0, 0))  
                
def create_gif(arrays, name, directory="files/gifs"):
    """
    Creates a gif out of a series of greyscale arrays
    
    Inputs:
        arrays, array: A list or array of (greyscale) arrays to turned into a gif
        name, String: Name of the gif to be created
        directory, string: location to place the gifs object in
        
    Returns:
        None
        
    Notes:
        Negative values are not handled properly! (can use np.abs() to fix for small values)
    """
    formatted_arrays = format_arrays(arrays)
    imageio.mimsave("{}/{}.gif".format(directory, name), formatted_arrays)
#     imageio.mimsave("{}/{}.gif".format(directory, name), arrays)
    
def format_arrays(arrays):
    """
    Formats (scales) the data in a series of arrays to be turned into gif series
    Primary use is to suspress warning output when creating a gif with imagieio
    """
    arrays_out = []
    for array in arrays:
        scaled_array = 255 * (np.abs(array) / np.max(array))
        formatted = scaled_array.astype(np.uint8)
        arrays_out.append(formatted)
        
    return arrays_out