from copy import deepcopy
import numpy as np

def get_Q(central, outer, mod=False):
    """
    Gets the "quality factor" defined as the ratio of the peak brightnesses of the central and outer region.
    A value closer to zero indicates that the peak brightnesses is similar in both regions
    A value above zero indicated a higher peak power in the outer region, below zero indicates higher peak in central region
    mod returns the absolute value of Q, |Q|
    """
    max_central = central.max()
    max_outer = outer.max()
    if not mod:
        return max_outer/max_central - 1
    return np.abs(max_outer/max_central - 1)
        
def test_quality(wf, npixels=1024, radius=10, print_Q=False, show=False):
    """
    Runs the FT and analysis on the wavefront
    Return the max of out outer region if the central peak brightness is less that the outer
    Else returns -1
    """
    
    FT = model_FT(wf, aperture*1e-3, npixels, [wl], fl, detector_pitch)
    central, outer = split_central(FT, npixels, radius)
    Q = get_Q(central, outer)
    if print_Q:
        print(print("Quality factor: {:.6f}".format(Q)))

    if show:
        display(wf, FT)
        display_split(FT, radius=radius)
        
    if Q < 0: # Is there a higher peak power in the central region of the detector than the outer ring?
        return -1, Q, FT # Disregard these results
    else:
        return outer.max(), Q, FT
    
def get_heuristics(population, ratio=0.05):
    pop = deepcopy(population)
    
    Qs = []
    Hs = []
    for i  in range(len(pop)):
        Qs.append(pop[i][1])
        Hs.append([pop[i][0],i])
        
    Qnorm = Qs/np.sum(Qs)
    
    # Hs[H, orig_index, ranked_index]
    
    Hs = sorted(Hs, key=itemgetter(0))
    for i in range(len(Hs)):
        Hs[i].append(i)
    Hs = sorted(Hs, key=itemgetter(1))
        
    vals = []
    for Q,H in zip(Qs, Hs):
        val = (np.log(Q)**0.5)*(len(Hs)-H[2])
        vals.append(val)
        
    valnorm = vals/np.sum(vals)
    
    for individual, val in zip(population,valnorm):
        individual.append(val)
    
    return population
        
    # The vals array now contains the new normalised heuristic evalution that takes into account both Q and H
    

