import numpy as np
import math
from math import sin, log10, cos, atan2, hypot

def spiral(r, phi, aperture, r_max, r_min, split_odd, split_even, first, second, third, fourth):
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

    if (r<=r_max and r>r_min):
        logr = log10(r)
        chi1 = alpha1*logr+m1*phi+eta1
        c1 = cos(chi1)
        chi2 = alpha2*logr+m2*phi+eta2
        c2 = cos(chi2)
        chi3 = alpha3*logr+m3*phi+eta3
        c3 = sin(chi3)
        
        if c3 >= 0:
            if sin(chi3/2.) > 0:                  # First  quadrant
                for i in range(len(odd)):
                    if first[i] == True and (r <= split_odd[i] and r > split_odd[i+1]):
                        v = black
                        return v
            else:                                 # Third quadrant
                for i in range(len(odd)):
                    if third[i] == True and (r <= split_odd[i] and r > split_odd[i+1]):
                        v = black
                        return v
                    
            v = black if (c1*c2*c3>0) else white
            return v
        
        else: # In the even regime
            if sin(chi3/2.) <= 0:                 # Second  quadrant
                for i in range(len(even)):
                    if second[i] == True and (r <= split_even[i] and r > split_even[i+1]):
                        v = black
                        return v
            else:                                 # Fourth quadrant
                for i in range(len(even)):
                    if fourth[i] == True and (r <= split_even[i] and r > split_even[i+1]):
                        v = black
                        return v
                
            v = black if (c1*c2*c3>0) else white
            return v

    elif r < r_min:
        v = black
    return v