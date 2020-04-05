import numpy as np
import imageio
from astropy.io import fits
from scipy.ndimage import zoom


# def pupil_from_fits(file_name, offset=0, output_size=0):
#     """
#     Takes in the fits file and returns a complex array of the pupil
#     """
#     # Create a fits object from astropy
#     fits_file = fits.open(file_name)[0].data
#     array = np.array(fits_file)
    
#     if output_size != 0:
#         ratio = output_size/array.shape[0]
#         scaled_array = zoom(array, ratio)

#         # Some values seem to get changed in the process, this is an ad-hoc fix
#         scaled_array[scaled_array >= np.pi] = np.pi
#         scaled_array[scaled_array < 0] = 0
        
#         array = scaled_array

#     # Calculate needed values
#     gridsize = array.shape[0] - 2*offset
#     c = gridsize//2
    
#     # Create value arrays
#     Xs = np.linspace(-c, c-1, num=gridsize)
#     X, Y = np.meshgrid(Xs, Xs)
#     r = np.hypot(X, Y)
    
#     # Create pupil
#     pupil = np.exp(1j*array)
    
#     # Zero outer regions
#     pupil[r >= (gridsize//2) + offset] = np.complex(0,0)
        
#     return pupil

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

def zemax_to_array(path_to_file):
    """
    Coverts Zemax .txt filed to numpy arrays
    """
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
    unit: unit of measurement {"mm": 0, "cm": 1, "in": 2, "m": 3}
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
    unit: unit of measurement {"mm": 0, "cm": 1, "in": 2, "m": 3}
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
    
    phases = []
    
    with open("{}.DAT".format(file_name), 'w') as f:
        f.write("{} {} {} {} {} {} {}\n".format(nx, ny, delx, dely, unitflag, xdec, ydec))
        
        for i in range(nx):
            for j in range(ny):
                phase_val = np.angle(pupil[i][j])
                
                if phase_val not in phases:
                    phases.append(phase_val)
                    
                if phase_val < 1e-12:
                    phase_val = 0
                f.write("{} {} {} {} {}\n".format(float(phase_val), 0, 0, 0, 0))  
    print(phases)
    
    
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