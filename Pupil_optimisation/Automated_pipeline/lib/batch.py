"""
Class for a batch object
"""

class Batch:
    def __init__(self, batch_name, batch_size):
    
    self.batch_name = batch_name
    self.batch_size = batch_size
    self.pupils = []
    
    pupil_inputs, simulation_settings = load_batch_data(os.getcwd())
    self.simulation_settings = simulation_settings
    self.pupil_inputs = pupil_inputs
    
    
    def __generate_initial_population(self):
        """
        Generates the random initial population
        """
        
        while len(self.pupils) < self.batch_size:
            pupil = generate_new_pupil(pupil_inputs, simulation_settings)
            self.pupils.append(pupil)
        
    def run_batch():
    
        iteration = 0
        initial_pupils = []
        
        while len(initial_pupils) < self.batch_size:
            pupil = generate_new_pupil(pupil_inputs, simulation_settings)
            
            initial_pupils.append(pupil)
            
            
        
        while True:
            if iteration < self.batch_size-1:

                if iteration%2 == 0:
                    pupil = generate_new_pupil(pupil_inputs, simulation_settings)            
                else:
                    pupil = modify_pupil(G_Saxburg_1_0, pupil, G_Saxburg_inputs, binary_backtransform_1_0) 

                initial_pupils.append(pupil)
#                 pupil.save_pupil_file(path)

                iteration += 1
#                 print(iteration)

            elif iteration == batch_size-1:
                # Generate new pupil
                pupil = generate_new_pupil(pupil_inputs, simulation_settings)
                initial_pupils.append(pupil)
                pupil.save_pupil_file(path)

                iteration += 1
                print(iteration)

                file_outputs = [[],[],[],[],[],[],[],[]]
                for file, key in zip(file_outputs[1:], pupil.heuristics.keys()):

                    if key == "PEAK" or key == "CENTRAL":
                        initial_pupils.sort(key=lambda x: x.heuristics[key])
                    else:
                        initial_pupils.sort(key=lambda x: x.heuristics[key], reverse=True)

                    for i in range(len(initial_pupils)):
                        file.append([i, initial_pupils[i].uid, initial_pupils[i].heuristics[key]])

                for i in range(len(initial_pupils)):
                    file_outputs[0].append([i, initial_pupils[i].uid, i])

                save_pupil_rankings(path, file_outputs)

            else:
                if iteration%2 == 0:
                    # Generate new pupil
                    pupil = generate_new_pupil(pupil_inputs, simulation_settings)
                else:
                    # Modify pupil
                    pupil = modify_pupil(G_Saxburg_1_0, pupil, G_Saxburg_inputs, binary_backtransform_1_0)

                # Evaluate pupil
                pupil.save_pupil_file(path)
                evaluate_pupil_ranking(path, pupil)
                iteration += 1
            print(iteration)
        
    
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

