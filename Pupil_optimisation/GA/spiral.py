import numpy as np
import math
from math import sin, log10, cos, atan2, hypot

def spiral(r, phi, aperture, r_max, r_min, split, first, second, third, fourth):
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
    
    white = np.complex(1,0)
    black = -np.complex(1,0)
    v = np.complex(0,0)
    offset = np.complex(-(3**0.5)/2,1/2)

    if (r<=r_max and r>r_min):
        logr = log10(r)
        chi1 = alpha1*logr+m1*phi+eta1
        c1 = cos(chi1)
        chi2 = alpha2*logr+m2*phi+eta2
        c2 = cos(chi2)
        chi3 = alpha3*logr+m3*phi+eta3
        c3 = sin(chi3)
        
        z = 0 if (c1*c2*c3>0) else 1 
        for i in range(len(split)):
            if (r <= split[i] and r > split[i+1]): # Finds which region we are in
                
                if i%2 != 0:
                    z = np.abs(z-1)
                
                # First quadrant
                if c3 < 0 and sin(chi3/2.) <= 0:
                    return black if first[i][z] else white

                # Second qudrant
                elif c3 >= 0 and sin(chi3/2.) <= 0:
                    return black if second[i][np.abs(z-1)] else white
#                     return black if second[i][z] else white
                
                # Third quadrant
                elif c3 < 0 and sin(chi3/2.) > 0:
                    return black if third[i][z] else white
        
                # Fourth qudrant
                else: 
                    return black if fourth[i][np.abs(z-1)] else white
#                     return black if fourth[i][z] else white
                    
    elif r < r_min:
#         v = black
        v = np.complex(0,0)
    return v