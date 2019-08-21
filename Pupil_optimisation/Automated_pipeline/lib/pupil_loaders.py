"""
A script to hold all the functions designed to load pupil objects from disk memory into a pupil object to be used
"""
from lib.pupil import Pupil

def load_pupil_file(path, uid):
    """
    Takes the pupil with uid located at path and returns a reconstructed pupil object
    """
    # Create the main pupil dictionary for returning
    dictionaries = []
    
    # Define the data types for each dictionary
    file_names = ["heuristics", "pupil_inputs", "simulation_settings"]
    heuristics_dtype = [float, float, float, float, float, float, float]
    pupil_inputs_dtype = [int, float, float, int, int, int, int, float]
    simulation_settings_dtype = [float, float, int, float, float, float]
    data_types = [heuristics_dtype, pupil_inputs_dtype, simulation_settings_dtype]
    
    # Open the file
    for file_name, data_type_list in zip(file_names, data_types):
        with open("{}/{}/{}.txt".format(path, uid, file_name), 'r') as f:
            dictionary = {}
            data = f.readlines()
            # Get the data formatted and put in the dictionary
            for line, dtype in zip(data, data_type_list):
                key, value = line.strip().split(": ")
                value_formatted = convert_data(value, dtype)
                dictionary[key] = value_formatted
                
        # Add dictionary to the primary dictionary
        dictionaries.append(dictionary)
        
    # Load the history file
    with open("{}/{}/{}.txt".format(path, uid, "history"), 'r') as f:
        key, value = line.strip().split(": ")
        value_formatted = convert_data(value, list)
        dictionaries.append(value_formatted)
    
    # Load the numpy files
    numpy_file_names = ["images", "pupil_outputs", "visual_analysis"]
    for file_name in numpy_file_names:
        dictionaries.append(load_from_npy(path, uid, file_name))

    # Unpack the relevent pupil values
    pupil_inputs = dictionaries[1]
    pupil_outputs = dictionaries[5]
    images = dictionaries[4]
    heuristics = dictionaries[0]
    simulation_settings = dictionaries[2]
    visual_analysis = dictionaries[6]
    history = dictionaries[3]
    
    # Construct the pupil object
    pupil_object = Pupil(pupil_inputs, pupil_outputs, images, heuristics, simulation_settings, visual_analysis, history)
    
    return pupil_object

def convert_data(data, dtype):
    """
    Converts the string data from text files to the approprite type
    Note:
        The list dtype is only designed to work for the history input and not others
    """
    if dtype == float:
        return float(data)
    elif dtype == int:
        return int(float(data))
    elif dtype == str:
        return str(data)
    elif dtype == list:
        # Note this is only designed to work for the history enrty and is NOT general
        if data == "[]":
            return []
        return list(data[1:-1].split(", "))
    else:
        raise TypeError("Data type {} input is not yet handeled by this function".format(dtype))

def load_from_npy(path, uid, file_name):
    """
    Loads the numpy files from the disk and returns numpys files formatted into dictionaries
    """
    from numpy import load

    file_names = ["images", "pupil_outputs", "visual_analysis"]
    category_names = [["pupil", "WF", "PSF"], 
                      ["num_radial_divisions", "radial_divisions", "num_angular_divisions", "angular_divisions"], 
                      ["regions", "peaks", "cum_sum"]]
    
    # Get the correct array names
    categories = category_names[file_names.index(file_name)]
    
    dictionary = {}
    for category in categories:
        dictionary[category] = load("{}/{}/{}/{}.npy".format(path, uid, file_name, category))
    
    return dictionary                