import sys
from operator import itemgetter
import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import random
path = str(Path().resolve())
sys.path.append((path[0:len(path)-len("notebooks/TinyTol")]) + "toliman-proper")
from math import sin, log10, cos, atan2, hypot
from FT_model import model_FT
from mutation import *
from evaluators import *
from removals import *
from generators import *
from visualise import *

def run_GA(pop_size, itters):
    population = generate_initial(pop_size)
    count = 0
    for i in range(itters):
        count += 1
        print("Iteration: {}".format(count))
        population = breed_population(population)
    return population 

def breed_population(population, num_children=3):
    pop = remove_null(population)
    pop = remove_duplicates(pop)
    num = num_children - len(pop)
    
    if num > 0:
        while len(pop) < num_children: # Keep generating children until we have a fit population of at least 3
            new_individual = generate_individual(None)
            if new_individual[0] != -1:
                pop.append(new_individual)
       
    population = get_heuristics(pop) # Move this into a generators function?           
                
    sorted_pop = sorted(population, key=itemgetter(5))
#     breeding_pop = sorted_pop[:3] # Take the top 3
    # Select breeding individuals based on probability
    
    breeding_pop = []
    index = 0
    while len(breeding_pop) < 3:
        if np.random.ranf(1)[0] < sorted_pop[index%len(sorted_pop)][5]:
            breeding_pop.append(sorted_pop[index%len(sorted_pop)])
        index += 1

    children = generate_children(breeding_pop)
    
    # Add freshly generated individuals
    
#     for individual in breeding_pop:
#         children.append(individual)
        
    population_out = remove_duplicates(children)
        
    return population_out

aperture = 0.015                 # Aperture (m)
gridsize = 2048
npixels = 1024                   # Size of detector, in pixels
wl = 0.525e-9                    # Wavelength values (micrometers)
fl = 0.15                        # Focal length (m)
detector_pitch = 1.12e-6         # m/pixel on detector (pixel spacing)

split_values = [15.15, 12.4, 10.17, 8.33, 6.83, 5.6, 3.75, 3.05, 2.5, 2.07, 1.7, 1.4, 1.14, 0.925, 0.76, 0.62]
r_max = split_values[0]
r_min = split_values[-1]

split_even =[15.5, 10.17, 6.83, 4.575, 3.05, 2.07, 1.4, 0.925, 0.62] 
second = [0,0,0,0,0,0,0,0]
fourth = [0,0,0,0,0,0,0,0]

split_odd = [15.5, 12.4, 8.33, 5.6, 3.75, 2.5, 1.7, 1.14, 0.76, 0.62]
first = [0,0,0,0,0,0,0,0,0]
third = [0,0,0,0,0,0,0,0,0]

splits = (split_odd, split_even)
settings = (first, second, third, fourth)

gridsize = 2048
wf = generate_spiral(gridsize, aperture*1e3, r_max, r_min, splits, settings)
print(test_quality(wf, print_Q=True, show=True, radius=0))

gridsize = 1024

pop = run_GA(3, 4)

for person in population:
#     print(person)
    print("Q: {:.5f}".format(person[1]))
    print("H: {:.5f}".format(person[0]*1e-8))
    print("Heuristic: {:.2f}".format(person[-1]))
    display(person[3], person[4])
    display_split(person[4], radius=10)
