"""
This script contains all of the methods of analysis fo the PSF. 
Many of the graident energy functions are as follows from the file "Documents/Some Gradient Energy Metrics.pdf"
All files that pertain to the analysis of the PSF should be placed in this script
"""
import numpy as np 
from math import sin, cos, atan2, hypot
    
def GE(array):
    """
    Gradient Energy
    """
    
    X,Y = np.gradient(array)
    out = np.zeros(array.shape)

    for i in range(len(array)):
        for j in range(len(array[0])):
            out[i][j] = (X[i][j]**2 + Y[i][j]**2)
            
    scaling_factor = 1e6
    return np.sum(out)*scaling_factor

def FTGE(array, max_radius):
    """
    Flat Topped Gradient Energy
    """
    X,Y = np.gradient(array)
    out = np.zeros(array.shape)
    c = array.shape[0]//2
    
    for i in range(len(array)):
        for j in range(len(array[0])):
            x = i - c
            y = j - c
            r = hypot(x, y)
            if r > max_radius:
                out[i][j] = 0
            else:
                out[i][j] = (X[i][j]**2 + Y[i][j]**2)
                
    scaling_factor = 1e6
    return np.sum(out)*scaling_factor

def RWGE(array):
    """
    Radially Weighted Gradient Energy
    """
    X,Y = np.gradient(array)
    out = np.zeros(array.shape)
    c = array.shape[0]//2

    for i in range(len(array)):
        for j in range(len(array[0])):
            x = i - c
            y = j - c
            out[i][j] = ( X[i][j]*cos(atan2(y,x)) + Y[i][j]*sin(atan2(y,x)) )**2
    
    scaling_factor = 1e6
    return np.sum(out)*scaling_factor

def FTRGE(array, max_radius):
    """
    Flat Topped Radial Gradient Energy
    """
    X,Y = np.gradient(array)
    out = np.zeros(array.shape)
    c = array.shape[0]//2

    for i in range(len(array)):
        for j in range(len(array[0])):
            x = i - c
            y = j - c
            r = hypot(x, y)
            
            if r > max_radius:
                out[i][j] = 0
            else:
                out[i][j] = ( X[i][j]*cos(atan2(y,x)) + Y[i][j]*sin(atan2(y,x)) )**2
    
    scaling_factor = 1e6
    return np.sum(out)*scaling_factor

def power_ratio(A, radius):
    """
    Determines the ratio of power inside the radius vs outisde the radius
    Assumes square inputs
    """    
    gridsize = A.shape[0]
    c = gridsize//2
    inner = 0
    
    for i in range(gridsize):
        for j in range(gridsize):
            x = i - c
            y = j - c
            r = hypot(x, y)
            if r < radius:
                inner += A[i][j]
                
    inner_percentage = inner/np.sum(A)
    return inner_percentage

def get_visual_analysis(PSF):
    """
    Gets various values as a function of radius for visual analysis
    Outputs:
        regions: The sum of power contained at some radii (function of radii)
        peaks: a 1D array representing the peak pixel power as a function of raidus
        cum_sum: Cumulative sum of power internal to some radii
    """    
    regions = np.zeros(PSF.shape[0])
    peaks = np.zeros(PSF.shape[0])
    cum_sum = np.zeros(PSF.shape[0])
    
    c = PSF.shape[0]//2
    for i in range(PSF.shape[0]):
        for j in range(PSF.shape[0]):
            x = i - c
            y = j - c
            r = int(hypot(x, y))
            regions[r] += PSF[i][j]
            peaks[r] = PSF[i][j] if PSF[i][j] > peaks[r] else peaks[r]
    
    for i in range(len(regions)):
        cum_sum[i] = regions[i] if i == 0 else cum_sum[i-1] + regions[i]
        
    return regions, peaks, cum_sum

