"""
Script containing the pupil object class and its related functions

Pupil class data structure:

    pupil_inputs - Dictionary
        array_size: int - Array size of simulation
        min_radius: float - Minimum radius of pupil
        max_radius: float - Maximum radius of pupil
        min_radial: int - Minimum number of radial divisions
        max_radial: int - Maximum number of radial divisions
        min_angular: int - Minimum number of angular divisions
        max_angular: int - Maximum number of radial divisions
        sampling: float - The sampling rate to convert number of pixels to distance (for min/max radius)

    pupil_outputs - Dictionary
        num_radial_divisions: int - Number of radial divisions chosen
        radial_divisions: list of floats - Values of said radial divisions
        num_angular_divisions: list of ints - Number of angular divisions chosen
        angular_divisions: 2D list of floats - values of angular divisions chosen

    images - Dictionary
        pupil: 2D Complex array
        WF: 2D complex array
        PSF: 2D float array

    heuristics - Dictionary
        RWGE: float
        FTRWGE: float
        GE: float
        FTGE: float
        RATIO: float
        PEAK: float
        CENTRAK: float

    visual_analysis - Dictionary
        regions: list of floats
        peaks: list of floats
        cum_sum: list of floats

    uid - 10 digit integer, "unique id"

    history - list of strings, represents the functions that have been applied since the pupil formulation so that it can be regenerated
"""
import os
from copy import deepcopy
from numpy.random import randint
from numpy import save

class Pupil:
    def __init__(self, pupil_inputs, pupil_outputs, images, heuristics, simulation_settings, visual_analysis, history):
    
        self.pupil_inputs = pupil_inputs
        self.pupil_outputs = pupil_outputs
        self.images = images
        self.heuristics = heuristics
        self.simulation_settings = simulation_settings
        self.visual_analysis = visual_analysis
        self.history = deepcopy(history)
        self.uid = randint(1e9, 1e10)

    def save_pupil_file(self, path):
        """
        Turns the pupil object data into the pre determined file structure
        Note: 
            Path needs to be the absolute path (using os.getcwd()) and end with a '/' to indicate its a directory
        """
        # Create the folders for the pupil data
        os.mkdir("{}/{}".format(path, self.uid))

        # Create the text files
        dictionaries = self.get_dictionaries()
        names = self.get_dictionary_names()
        uid = self.uid

        # Create the heuristics file
        for dictionary, name in zip(dictionaries, names):
            file_path = "{}/{}/{}.txt".format(path, uid, name)
            self.write_dictionary_to_file(dictionary, file_path)

        # Create history file
        file_path = "{}/{}/{}.txt".format(path, uid, "history")
        with open(file_path, 'w') as f:
            f.write(str(self.history))

        # Write numpy data to files
        numpy_file_names = self.get_numpy_dictionary_names()
        numpy_dictionaries = self.get_numpy_dictionaries()

        # Create the numpy files
        for dictionary, name in zip(numpy_dictionaries, numpy_file_names):
            os.mkdir("{}/{}/{}/".format(path, uid, name))
            file_path = "{}/{}/{}/".format(path, uid, name)
            self.save_as_npy(dictionary, file_path)
            
    def get_dictionaries(self):
        return self.pupil_inputs, self.heuristics, self.simulation_settings
    
    def get_dictionary_names(self):
        return "pupil_inputs", "heuristics", "simulation_settings"
    
    def get_numpy_dictionaries(self):
        return self.images, self.pupil_outputs, self.visual_analysis
    
    def get_numpy_dictionary_names(self):
        return "images", "pupil_outputs", "visual_analysis"
            
    def write_dictionary_to_file(self, dictionary, file):
        """
        Writes a dictionary to a file with a specified header
        Note:
            'file' needs to conatin the absolute path to the file and its data type (ie .txt) in the string
        """
        with open(file, 'w') as f:
            for key in dictionary.keys():
                f.write("{}: {}\n".format(key, dictionary[key]))

    def save_as_npy(self, dictionary, path):
        """
        Saves the image files in the numpy binary format to the specified path
        """
        for key in dictionary.keys():
            save(path + key, dictionary[key])