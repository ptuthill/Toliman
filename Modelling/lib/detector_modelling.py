import numpy as np
import imageio

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
    
def format_arrays(arrays):
    """
    Formats (scales) the data in a series of arrays to be turned into gif series
    Primary use is to suspress warning output when creating a gif with imagieio
    """
    arrays_out = []
    for array in arrays:
        scaled_array = 255 * (array / np.max(array))
        formatted = scaled_array.astype(np.uint8)
        arrays_out.append(formatted)
        
    return arrays_out