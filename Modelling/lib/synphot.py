import numpy as np
import pysynphot as S

def create_stellar_obs(aperture, central_obscuration, nwavels, wl_range, star_dict):
    """
    Creates an observation object from pysynphot phoenix models
    
    Parameters:
        Aperture (cm): Aperture of the telescope
        central_obscuration (cm): diameter of central obscuration of telescope
        nwaels: number of wavelengths to sample
        wl_range (Angstroms): [first wavelength, last wavelength] 
        star_dict: dictionary of the following structure
            {"mag": float, # vega mag
            "Teff": int,
            "Z": float,
            "log g": float}
            
    Returns:
        A pysynphot observation object describing the star being observed through the given telescope architecture 
    """
    
    # Set Telescope values
    r = (aperture-central_obscuration)/2
    collecting_area = (np.pi * r**2)
    S.refs.setref(area=collecting_area) # Takes units of cm^2
    wavels = np.linspace(wl_range[0], wl_range[1] ,nwavels)
    throughput = np.ones(nwavels)
    bandpass = S.ArrayBandpass(wavels, throughput)
    
    # Create star object
    star_obj = S.Icat('phoenix', star_dict["Teff"], star_dict["Z"], star_dict["log g"])
    spec_filt = star_obj.renorm(star_dict["mag"], 'vegamag', bandpass)
    
    # Create observation object
    obs = S.Observation(spec_filt, bandpass, binset=wavels)
    
    return obs