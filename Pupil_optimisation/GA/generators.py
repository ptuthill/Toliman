import numpy as np
import mutation
from spiral import spiral
from evaluators import test_quality

def generate_initial(pop_size):
    population = []
    while len(population) < pop_size:
        individual = generate_individual(None)
        if individual[0] != -1:
            population.append(individual)
    return population

def generate_individual(settings):
    
    wf = generate_spiral(gridsize, aperture*1e3, r_max, r_min, splits, settings)
    H, Q, FT = test_quality(wf)

    return [H, Q, settings, wf, FT]


def generate_children(population):
    pair1 = [population[0][2], population[1][2]]
    pair2 = [population[1][2], population[2][2]]
    pair3 = [population[2][2], population[0][2]]
    
    pairs = [pair1, pair2, pair3]
    children = []
    for pair in pairs:
        c1,c2 = swap_rows(pair)
        c3,c4,c5,c6 = swap_ends(pair)
        children += c1,c2,c3,c4,c5,c6
        
#     pop = mutate(children)
    pop = children
    
    new_pop = []
    for child in pop:
        new_pop.append(generate_individual(child))

    return new_pop
        
def generate_settings(prob=0.9):
    """
    Individuals are skewed towards to having more empty spaces than filled ones
    If the generated individual has a Q < 1, it is disregarded and a new one is made #CHANGED#
    """
    first = np.random.choice(a=[0, 1], size=(1, 8), p=[prob, 1-prob])[0]
    third = np.random.choice(a=[0, 1], size=(1, 8), p=[prob, 1-prob])[0]
    second = np.random.choice(a=[0, 1], size=(1, 9), p=[prob, 1-prob])[0]
    fourth = np.random.choice(a=[0, 1], size=(1, 9), p=[prob, 1-prob])[0]
    settings = [first, second, third, fourth]

def generate_spiral(gridsize, aperture, r_max, r_min, splits, settings):
    split_odd = splits[0]
    split_even = splits[1]
    first = settings[0]
    second = settings[1]
    third = settings[2]
    fourth = settings[3]
    
    sampling = aperture/(gridsize//2)
    wfarr = np.zeros([gridsize, gridsize], dtype = np.complex128)
    c = gridsize//2
    for i in range(gridsize):
        for j in range(gridsize):
            x = i - c
            y = j - c
            phi = math.atan2(y, x)
            r = sampling*math.hypot(x,y)
            wfarr[i][j] = spiral(r, phi, aperture, r_max, r_min, split_odd, split_even, first, second, third, fourth)
    return wfarr

