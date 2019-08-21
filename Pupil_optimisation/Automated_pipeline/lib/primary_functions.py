"""
Script for all the main functions that are used by driver.py and the notebook
Most functions here are high level and call on the other scripts in lib/
"""
from lib.secondary_functions import *
from lib.ranking_functions import *
from lib.WF_modifiers import *
from lib.backtransformers import *
from lib.PSF_analysis_functions import *
from lib.pupil_loaders import *
from lib.pupil import Pupil
import os
import numpy as np
from numpy import linspace
from numpy.random import randint, random_sample
from math import hypot, atan2

def load_batch_data(path):
    """
    Loads the data stores in "pupil_inputs.txt" and "simulation_settings.txt" to be used as the inputs for a given batch rn
    """
    file_names = ["pupil_inputs", "simulation_settings"]
    pupil_inputs_dtype = [int, float, float, int, int, int, int, float]
    simulation_settings_dtype = [float, float, int, float, float, float]
    data_types = [pupil_inputs_dtype, simulation_settings_dtype]
    
    # Open the file
    dictionaries = []
    for file_name, data_type_list in zip(file_names, data_types):
        with open("{}/{}.txt".format(path, file_name), 'r') as f:
            dictionary = {}
            data = f.readlines()
            # Get the data formatted and put in the dictionary
            for line, dtype in zip(data, data_type_list):
                key, value = line.strip().split(": ")
                value_formatted = convert_data(value, dtype)
                dictionary[key] = value_formatted
                
        # Add dictionary to the primary dictionary
        dictionaries.append(dictionary)
    
    return dictionaries[0], dictionaries[1]

def generate_new_pupil(pupil_inputs, simulation_settings):
    """
    Generates a new pupil from the pupil inputs dictionary and creates a pupil object by generating the images and analysis etc 
    """
    pupil_array, pupil_outputs, history = generate_pupil(pupil_inputs)
    images, heuristics, visual_analysis  = evaluate_pupil(pupil_array, simulation_settings)
    pupil_object = Pupil(pupil_inputs, pupil_outputs, images, heuristics, simulation_settings, visual_analysis, history)

    return pupil_object

def modify_pupil(mod_function, pupil_object, function_inputs, transform_function):
    """
    Modifies the given pupil with the mod function and given inputs
    function inputs should be a dictionary formatted correctly for the given mod_functions
    """
    # Unpack values
    simulation_settings = pupil_object.simulation_settings
    pupil_inputs = pupil_object.pupil_inputs
    pupil_outputs = pupil_object.pupil_outputs
    
    # Modify WF
    modified_WF, history = mod_function(pupil_object, function_inputs)
    # Backtransform WF
    modified_pupil, history = transform_function(pupil_object, history, modified_WF)
    # Evaluate new pupil
    images, heuristics, visual_analysis  = evaluate_pupil(modified_pupil, simulation_settings)
    # Construct new pupil
    new_pupil = Pupil(pupil_inputs, pupil_outputs, images, heuristics, simulation_settings, visual_analysis, history)
        
    return new_pupil

def generate_pupil(input_values):
    """
    Generates a randomly initialised pupil based on the input values
    Inputs: 
        input_values (dictionary):
            array_size, int: Size of array in pixels
            mix/max radius, float: Minimum and maximum physical radius values of the aperture
            mix/max radial, int: Minimium and maximum number of radial regions
            min/max angular, int: Minimum and maximum angular division per radial division
    Outputs:
        output_values (dictionary):
            radial_divisions, list of floats: The radail values used to generate the pupil
            angular divisions, 2D list of floats: a list of The angular values used to generte the pupils
        pupil_array, 2d complex array: Complex array representing the phase pupil
    """
    # Get values from inputs
    array_size = input_values["array_size"]
    min_radius = input_values["min_radius"]
    max_radius = input_values["max_radius"]
    min_radial = input_values["min_radial"]
    max_radial = input_values["max_radial"]
    min_angular = input_values["min_angular"]
    max_angular = input_values["max_angular"]
    
    # Create output dictionary
    output_values = {}
    
    # Generate radial divisions
    num_radial_regions = randint(low=min_radial, high=max_radial)
    radial_divisions = sorted((max_radius - min_radius)*random_sample((num_radial_regions,)) + min_radius)
    output_values["num_radial_divisions"] = num_radial_regions
    output_values["radial_divisions"] = radial_divisions

    # Generate angular divisions
    angular_divisions = []
    num_angular_divisions_list = []
    for i in range(num_radial_regions):
        num_angular_divisons = randint(low=min_angular, high=max_angular)*2
        num_angular_divisions_list.append(num_angular_divisons)
        offset = 2*np.pi*random_sample((1,))[0]
        angular_values = sorted((linspace(0, 2*np.pi, num=num_angular_divisons+1)[:-1] + offset)%(2*np.pi) - np.pi)
        angular_divisions.append(angular_values)
    output_values["num_angular_divisions"] = num_angular_divisions_list
    output_values["angular_divisions"] = angular_divisions
        
    # Assign values to output array
    c = array_size//2
    sampling = 1/(c/max_radius)
    input_values["sampling"] = sampling
    
    array = np.zeros([array_size, array_size], dtype=complex)
    for i in range(array_size):
        for j in range(array_size):
            x = i-c
            y = j-c

            # Find the radial region we are in
            r = sampling*hypot(x, y)
            for radial_index in range(len(radial_divisions)-1, -1, -1):
                if r >= radial_divisions[radial_index]:
                    break

            # Find the angular region we are in
            angular_division = angular_divisions[radial_index]
            phi = atan2(x, y)
            for angular_index in range(len(angular_divisions[radial_index])):
                if phi >= angular_division[angular_index] and phi < angular_division[(angular_index+1)%len(angular_division)]:
                    break

            # Assign relevant values
            if r > max_radius or r < min_radius:
                array[i][j] = np.complex(0,0)
            elif angular_index%2 == 1:
                array[i][j] = np.complex(1,0)
            else:
                array[i][j] = -np.complex(1,0)
    
    # Create an empty history
    history = []
                
    return array, output_values, history

def evaluate_pupil(pupil, input_values):
    """
    Performs the analysis of the given pupil to be used to create a pupil object
    
    Inputs:
        pupil, 2D complex array:              Complex array representing the phase pupil
        input_values (dictionary):
            aperture, float (m):              Telescope aperture size in meters
            focal_length, float (m):          Focal length of the telescope in meters
            detector_size, int (pix):         Size of the detector in pixels
            detector_pitch, float (m):        Pitch of the detector in m (spacing of the unit cells)
            wavelength, float (m):            Wavelength that we are simulating through the telescope in meters
            fringe_extent, float (unitless):  The fringe radius that is to be used to evalute the pupil
    Outputs:
        images (dictionary):
            pupil, 2D complex array:  Array representing the phase pupil
            WF, 2D complex array:     Array representing the eletric field at the detector
            PSF, 2D float array:      Array representing the image formed at the detector
        heuristics (dictionary):
            RWGE, float:              Radially weighted gradient energy
            FTRWGE, float:            Flat topped radially weighted gradient energy
            GE, float:                Gradient energy
            FTGE, float:              Flat topped gradient energy
            ratio, float:             Ratio of power inside the fringe_extent vs outside (given as percentage of inner to outer)
            peak, float:              Peak pixel value of the PSF
            central, float: central   Pixel value of the PSF
        visual_analysis (dictionary):
            regions, 1D float array:  Peak pixel value as a function of radius
            peaks, 1D float array:    Total percentage of power contained at some radial region 
            cum_sum, 1D float array:  Total internal power to some radius
    Notes:
        At this point this should only work with monochromatic analysis since the fringe radius in pixels is calcuated every time, ie treated as ideailsed
    """    
    # Get values from inputs
    aperture = input_values["aperture"]
    focal_length = input_values["focal_length"]
    detector_size = input_values["detector_size"]
    detector_pitch = input_values["detector_pitch"]
    wavelength = input_values["wavelength"]
    fringe_extent = input_values["fringe_extent"]
    
    # Create output dictionaries
    images = {}
    heuristics = {}
    visual_analysis = {}

    # Calculate values needed for analysis
    c = detector_size//2
    pixel_radii = fringes_to_pixels(fringe_extent, aperture, focal_length, wavelength, detector_pitch)
    
    # Save pupil array
    images["pupil"] = pupil

    # Generate normalised WF
    WF = model_FT(pupil, aperture, detector_size, [wavelength], focal_length, detector_pitch, power=False)
    WF = WF/np.sum(WF) # Normalise
    images["WF"] = WF
    
    # Generate normalised PSF
    PSF = np.abs(WF)**2 
    PSF = PSF/np.sum(PSF) # Normalise
    images["PSF"] = PSF
        
    # Calculate various gradient energy values as defined by kieran in the "Some Gradient Energy Metrics.pdf"
    # Radially Weighted Gradient Energy
    rwge = RWGE(PSF)
    heuristics["RWGE"] = rwge
    
    # Flat Topped Radially Weighted Gradient Energy    
    ftrwge = FTRGE(PSF, max_radius=pixel_radii)
    heuristics["FTRWGE"] = ftrwge
    
    # Greadient Energy
    ge = GE(PSF)
    heuristics["GE"] = ge
    
    # Flat Topped Gradient Energy
    ftge = FTGE(PSF, max_radius=pixel_radii)
    heuristics["FTGE"] = ftge 
    
    # % Power contained within fringe extent
    ratio = power_ratio(PSF, pixel_radii)*100
    heuristics["RATIO"] = ratio
    
    # Peak pixel value
    pixel_peak = np.max(PSF)
    heuristics["PEAK"] = pixel_peak
    
    # Central pixel value
    central = PSF[PSF.shape[0]//2][PSF.shape[0]//2]
    heuristics["CENTRAL"] = central
    
    # Get data for graphing
    regions, peaks, cum_sum = get_visual_analysis(PSF)
    visual_analysis["regions"] = np.array(regions)
    visual_analysis["peaks"] = np.array(peaks)
    visual_analysis["cum_sum"] = np.array(cum_sum)
            
    return images, heuristics, visual_analysis

def create_new_batch(batch_name, simulation_settings):
    """
    Creates the file structure for a new batch of pupil generation
    Returns the absolute path to the batch
    Note: 
        The generation of the heurstic files in the root 'batch name' folder does NOT use absolute path and so may fail
    """    
    # Check to make sure we aren't overwriting any batches
    if os.path.isdir(batch_name):
        raise KeyError("Batch with name \'{}\' already exists".format(batch_name))
    else:
        os.mkdir(batch_name)
        
    # Create batch info file
    with open("{}/batch_info.txt".format(batch_name), 'w') as f:
        for key in simulation_settings.keys():
            f.write("{}: {}\n".format(key, simulation_settings[key]))
        f.write("\nNote: all units in meters")
    
    # Get path to batch directory to return
    absolute_path = os.getcwd() + '/' + batch_name + '/'
    return absolute_path

