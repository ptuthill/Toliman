import numpy as np

def arcsec_to_pixel(azimuthal_offset, pixel_size, focal_length):
    """
    Converts an 'azimuthal offset' in arcseconds into a pixel value 
    Inputs:
        azimuthal_offset, arcsec: Angular offset from centre point of FoV
        pixel_size, m: Physical size of pixels in array
        focal_length, m: Focal length of the telescope
    """
    return azimuthal_offset / rad_to_arcsec(pixel_size / focal_length)

# From degrees
def deg_to_rad(angle):
    return angle * np.pi / 180 

def deg_to_arcmin(angle):
    return angle * 60

def deg_to_arcsec(angle):
    return angle * 3600


# From radians
def rad_to_deg(angle):
    return angle * 180 / np.pi

def rad_to_arcmin(angle):
    return angle * (60 * 180) / np.pi

def rad_to_arcsec(angle):
    return angle * 3600 * 180 / np.pi


# From arcminutes
def arcmin_to_rad(angle):
    return angle * np.pi / (60 * 180)

def arcmin_to_deg(angle):
    return angle / 60

def arcmin_to_asec(angle):
    return angle * 60


# From arcseconds
def arcsec_to_rad(angle):
    return angle * np.pi / (180 * 3600)

def arcsec_to_deg(angle):
    return angle / 3600

def arcsec_to_arcmin(angle):
    return angle / 60
