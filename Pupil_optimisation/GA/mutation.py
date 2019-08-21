import numpy as np

def mutate(population, variance=0.1, prob=0.1):
    for individual in population:
        if np.random.ranf(1)[0] <= variance:
            row = np.random.randint(4)
            index = np.random.randint(len(individual[row]))
            if individual[row][index] == 1:
                individual[row][index] = 0
            else: 
                if np.random.ranf(1)[0] <= 1-prob: # Only flip a 0 to a 1 probabilistically
                    individual[row][index] = np.abs(individual[row][index]-1)
    return population

def swap_rows(pair):
    child1 = pair[0]
    child2 = pair[1]
    c1 = [child1[0], child1[1], child2[2], child2[3]] # Take the first and fourth from child 1, second and third from child 2
    c2 = [child2[0], child2[1], child1[2], child1[3]]
    return c1,c2

def swap_ends(pair):
    child1 = pair[0]
    child2 = pair[1]
    c1 = [np.concatenate((child1[0][:5],child2[0][5:])), np.concatenate((child1[1][:5],child2[1][5:])), child1[2], child1[3]]
    c2 = [child1[0], child1[1], np.concatenate((child1[2][:5],child2[2][5:])), np.concatenate((child1[3][:5],child2[3][5:]))]
    c3 = [np.concatenate((child2[0][:5],child1[0][5:])), np.concatenate((child2[1][:5],child1[1][5:])), child2[2], child1[3]]
    c4 = [child2[0], child1[1], np.concatenate((child2[2][:5],child1[2][5:])), np.concatenate((child2[3][:5],child1[3][5:]))]
    return c1,c2,c3,c4