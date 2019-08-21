import numpy as np

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
def asec_to_rad(angle):
    return angle * np.pi / (180 * 3600)

def arcsec_to_deg(angle):
    return angle / 3600

def arcsec_to_arcmin(angle):
    return angle / 60