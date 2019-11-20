import os
from lib.primary_functions import *

# Inputs
batch_name = "test3"
batch_size = 25
G_Saxburg_inputs = {"outer_limit": 10, "inner_limit": 0.55}


pupil_inputs, simulation_settings = load_batch_data(os.getcwd())
path = create_new_batch(batch_name, simulation_settings)
print(path)
iteration = 0
initial_pupils = []
while True:
    if iteration < batch_size-1:
        
        if iteration%2 == 0:
            pupil = generate_new_pupil(pupil_inputs, simulation_settings)            
        else:
            pupil = modify_pupil(G_Saxburg_1_0, pupil, G_Saxburg_inputs, binary_backtransform_1_0) 
            
        initial_pupils.append(pupil)
        pupil.save_pupil_file(path)
        
        iteration += 1
        print(iteration)
        
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
        