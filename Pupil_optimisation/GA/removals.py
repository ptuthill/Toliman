def remove_null(population):
    successors = []
    for individual in population:
        if individual[0] != -1:
            successors.append(individual)
    return successors

def remove_duplicates(pop):
    out = []
    vals = []
    for individual in pop:
        if individual[0] not in vals:
            vals.append(individual[0])
            out.append(individual)
            