import numpy as np
from math import sin, log10, cos, atan2, hypot

def binarized_ringed(r, phi, phase, thresh=0., white=0, empty=0.):
    # Spiral parameters
    alpha1 = 20.186
    m1 = 5
    eta1 = -1.308
    m2 = -5
    alpha2 = 16.149
    eta2 = -0.733
    m3 = 10
    alpha3 = 4.0372
    eta3 = -0.575    

    # Calculations assume a space with fixed radii 
    # parameters. These can be scaled to physical units
    scale = 0.15/300. # m/internal sampling dist
    # Physical dimensions
    r_max = 300.
    r_min = 50.
    r_split = 246. # interface between main spiral and outer rim    
    
    black = phase
    v = empty
    r = r/scale
    if (r<=r_max and r>r_min):
        logr = log10(r)
        c1 = cos(alpha1*logr+m1*phi+eta1)
        c2 = cos(alpha2*logr+m2*phi+eta2)
        c3 = sin(alpha3*logr+m3*phi+eta3)
        if (r>r_split): # Outer rim
            v=white if (c3<thresh or c1*c2*c3>thresh) else black
        else: # Main spiral
            v=white if (c1*c2*c3>thresh) else black
    return v

def binarized_ringed_flipped(r, phi, phase, thresh=0., white=0, empty=0., r_max=300., r_min=50., r_split=246.):
    # Spiral parameters
    alpha1 = 20.186
    m1 = 5
    eta1 = -1.308
    m2 = -5
    alpha2 = 16.149
    eta2 = -0.733
    m3 = 10
    alpha3 = 4.0372
    eta3 = -0.575    

    s = 0.15/300. # m/internal sampling dist
    # Physical dimensions
#     r_max = 300.
#     r_max = 10.
#     r_min = 50.
#     r_min = 0.
#     r_split = 246. # interface between main sprial and outer rim
#     r_split = 246./30.
    
    black = phase
    v = empty
    r = r/s
    if (r<=r_max and r>r_min):
        logr = log10(r)
        chi1 = alpha1*logr+m1*phi+eta1
        c1 = cos(chi1)
        chi2 = alpha2*logr+m2*phi+eta2
        c2 = cos(chi2)
        chi3 = alpha3*logr+m3*phi+eta3
        c3 = sin(chi3)
        if (r>r_split): # Outer rim
            if (c3<thresh):
                if (sin(chi3/2.)<thresh):
                    v=black if (c1*c2*c3>thresh) else white
                else:
                    v=black                
            else:                
                v=black if (c1*c2*c3>thresh) else white
        else: # Main spiral
            v=black if (c1*c2*c3>thresh) else white
    return v