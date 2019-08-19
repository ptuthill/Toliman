import numpy as np

def get_red(image):
    """
    Returns the red channel from the RGGB bayer pattern
    Assumes and outputs a square image
    """
    chip_dim = image.shape[1]
    bayer_red = np.remainder((np.arange(chip_dim)),2)
    bayer_red = np.outer(bayer_red,bayer_red)
    im_red = np.zeros((chip_dim//2,chip_dim//2))
    im_red = (image[bayer_red.nonzero()]).reshape(chip_dim//2,chip_dim//2)
    return im_red

    
def get_green(image):
    """
    Returns the green channel from the RGGB bayer pattern
    Assumes and outputs a square image
    Returned image is rotated by 45 degrees and has null output for the corners
    """
    chip_dim = image.shape[1]
    bayer_green = np.remainder((np.arange(chip_dim**2))+np.arange(chip_dim**2)/chip_dim,2)
    bayer_green = bayer_green.reshape(chip_dim,chip_dim)
    gnz = bayer_green.nonzero() 
    g_mapping = [chip_dim//2 -1 - (gnz[0]-gnz[1])//2,(gnz[0]+gnz[1])//2]
    im_green=np.zeros((chip_dim,chip_dim))
    im_green[g_mapping] = image[gnz]
    return im_green
    
    
def get_blue(image):
    """
    Returns the blue channel from the RGGB bayer pattern
    Assumes and outputs a square image
    """
    chip_dim = image.shape[1]
    bayer_blue = np.remainder((np.arange(chip_dim))+1,2)
    bayer_blue = np.outer(bayer_blue,bayer_blue)
    im_blue = np.zeros((chip_dim//2,chip_dim//2))
    im_blue = (image[bayer_blue.nonzero()]).reshape(chip_dim//2,chip_dim//2)
    return im_blue