import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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

pupil = pupil_from_fits("pupil_file.fits")

plt.imshow(np.angle(pupil))
plt.title("Phase change induced by pupil, units of phase")
plt.colorbar()
plt.show()