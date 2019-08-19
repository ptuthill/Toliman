import numpy as np
import pysynphot as S

def custom_bandpass(wl, pass_range, value=1):
    """
    Outputs a scaled custom binary bandpass
    """
    bandpass = []
    for wave in wl:
        if wave >= pass_range[0] and wave <= pass_range[1]:
            bandpass.append(value)
        else:
            bandpass.append(0)
    return bandpass

def create_stars(star_list):
    """
    Creates a list of star objects, out of either MizarA, MizarB, and Alcor
    """
    
    r = 1 # cm
    tinytol_area = np.pi*r**2
    S.refs.setref(area=tinytol_area)  # cm2

    nwavels = 3501
    wavels = np.linspace(3500,7000,nwavels)
    
    stars = []

    if "MizarA" in star_list:
        MizarA  = S.Icat('phoenix',9340,0.015,3.88)
        # logg
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2009A%26A...501..297Z
        # Temp 
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2009A%26A...501..297Z
        specMA = MizarA.sample(wavels)
        stars.append([MizarA,specMA,"MizarA",2.23])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002A%26A...384..180F

        
    if "MizarB" in star_list:
        MizarB  = S.Icat('phoenix',8280,0.015,3.88)
        # Temp 
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2012A%26A...537A.120Z
        # logg
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2009A%26A...501..297Z
        specMB = MizarB.sample(wavels)
        stars.append([MizarB,specMB,"MizarB",3.88])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002A%26A...384..180F

        
    if "Alcor" in star_list:
        Alcor  = S.Icat('phoenix',7955,0.015,3.88)
        # temp and logg
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2003AJ....126.2048G
        specA = Alcor.sample(wavels)
        stars.append([Alcor,specA,"AlcorA",4.01])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=1993A%26AS..100..591O

        
    if "Tauri_25" in star_list:
        T_25  = S.Icat('phoenix',13490,0.015,4.0) 
        # Temp logg
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2001A%26A...378..861C
        specT_25 = T_25.sample(wavels)
        stars.append([T_25,specT_25,"Tauri_25",2.87])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D


    if "Tauri_27" in star_list:
        T_27  = S.Icat('phoenix',13020,0.015,4.1)
        # Only Temp
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2009A%26A...501..297Z
        specT_27 = T_27.sample(wavels)
        stars.append([T_27,specT_27,"Tauri_27",3.62])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D


    if "Tauri_17" in star_list:
        T_17  = S.Icat('phoenix',14655,0.015,3.03)
        # Temp Log G quoted as 3.03??? 
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2001A%26A...378..861C
        specT_17 = T_17.sample(wavels)
        stars.append([T_17,specT_17,"Tauri_17",3.70])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D


    if "Tauri_20" in star_list:
        T_20  = S.Icat('phoenix',14310,0.015,3.5)
        # Temp 
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2009A%26A...501..297Z
        # log g 3.5 newest source is 
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=1979ApJS...41..675H
        specT_20 = T_20.sample(wavels)
        stars.append([T_20,specT_20,"Tauri_20",3.87])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D

    if "Tauri_23" in star_list:
        T_23  = S.Icat('phoenix',14521,0.015,2.92)
        # Temp log g 2.92??? Check units maybe?
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2001A%26A...378..861C
        specT_23 = T_23.sample(wavels)
        stars.append([T_23,specT_23,"Tauri_23",4.18])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D


    if "Tauri_19" in star_list:
        T_19  = S.Icat('phoenix',14180,0.015,4.0)
        # Temp
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2009A%26A...501..297Z
        specT_19 = T_19.sample(wavels)
        stars.append([T_19,specT_19,"Tauri_19",4.30])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D

        
    if "Tauri_28" in star_list:
        T_28  = S.Icat('phoenix',13366,0.015,3.82)
        # Temp and log g
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2001A%26A...378..861C
        specT_28 = T_28.sample(wavels)
        stars.append([T_28,specT_28,"Tauri_28",5.09])
        # mags
        # http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2002yCat.2237....0D

        
    return stars
        
    
def apply_filter(stars, colour):
    """
    Generates a binary filter based on the colour selected and passes the stars 
    spectrum through this filter, being normalised to its actual magnitude as observed
    outputs an oberservation from this spectrum for use in other programs
    """
    
    nwavels = 3501
    wavels = np.linspace(3500,7000,nwavels)

    if colour == "red":
        throughput = custom_bandpass(wavels, [5750,6500]) # 750nm range
    elif colour == "green":
        throughput = custom_bandpass(wavels, [4750,6000]) # 1250nm range
    elif colour == "blue":
        throughput = custom_bandpass(wavels, [4000,5250]) # 1250nm range

    band_bin = S.ArrayBandpass(wavels, throughput)
    
    for star in stars:
        star_obj = star[0]
        mag = star[3]

        filtered_spec = star_obj.renorm(mag,'vegamag',band_bin)
        observation = S.Observation(filtered_spec,band_bin,binset=wavels)

        star.append(filtered_spec)
        star.append(observation)
        
    return stars
        


def get_spec_even(star_list, colour, spacing, microns=False):
    """
    star_list: list of some combination of "MizarA", "MizarB", and "Alcor"
    colour: either "red" OR "green" OR "blue"
    spacing: Spacing between the wavelengths
    return a list of spectrums and their relative weights(fluxes), and wavelengths in m
    microns input allows for wavelength unit output to be converted to microns
    """
    
    stars = create_stars(star_list)
    stars = apply_filter(stars, colour)
    
    if microns == True:
        scale = 1e4
    else:
        scale = 1e10
    
    output_spec = []
    for star in stars:
        obs = star[5]
        waves = obs.wave
        weights = obs.flux
        wave_list = []
        weight_list = []
        for i in range(len(waves)):
            if weights[i] != 0 and waves[i]%spacing == 0:
                wave_list.append(waves[i]/scale)
                weight_list.append(weights[i])
        output_spec.append((wave_list,weight_list))

    return output_spec

def get_spec(star_list, colour, samples, microns=False):
    """
    star_list: list of some combination of "MizarA", "MizarB", and "Alcor"
    colour: either "red" OR "green" OR "blue"
    spacing: Spacing between the wavelengths
    return a list of spectrums and their relative weights(fluxes), and wavelengths in m
    microns input allows for wavelength unit output to be converted to microns
    """
    
    
    stars = create_stars(star_list)
    stars = apply_filter(stars, colour)
    
    if microns == True:
        scale = 1e4
    else:
        scale = 1e10
        
    if colour == "red":
        wave_range = np.linspace(5750,6500,751)
        num = 751
        
    elif colour == "green":
        wave_range = np.linspace(4750,6000,1251)
        num = 1251
        
    elif colour == "blue":
        wave_range = np.linspace(4000,5250,1251)
        num = 1251
        
    get_index = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    index = get_index(samples, num)
    wavels = [wave_range[i] for i in index]
        
    output_spec = []
    for star in stars:
        obs = star[5]
        waves = obs.wave
        weights = obs.flux
        wave_list = []
        weight_list = []
        for wave in wavels:
            wave_list.append(wave/scale)
            weight_list.append(weights[np.where(waves==wave)[0][0]])
        output_spec.append((wave_list,weight_list))

    return output_spec


def get_counts(star_list, colour, fps=10):
    """
    star_list: list of some combination of "MizarA", "MizarB", and "Alcor"
    colour: either "red" OR "green" OR "blue"
    fps: frames per second
    return count rates per frame, scaled as per the bayer pattern (green = 1/2 of total, red/blue = 1/4 of total)
    """
    
    stars = create_stars(star_list)
    stars = apply_filter(stars, colour)

    for star in stars:
        obs = star[5]        
        obs.primary_area = np.pi**2
        counts = obs.countrate()
        frame_count = counts/fps
        
    #if colour == "green":
    #    return frame_count/2
    #
    #return frame_count/4
    return frame_count
