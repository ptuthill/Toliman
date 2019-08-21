"""
A script to house all the functions related to assesing and maintaing the rankings of the current pupils in circulation
"""
import numpy as np
from shutil import rmtree

def evaluate_pupil_ranking(path, pupil_object):
    """
    Evaluates the ranking of the pupil object to the rest of the pupils currently in the circulation
    The function is well commented in order to make the following of the logic easier
    """
    # Load the data from the files
    rankings, RWGE, FTRWGE, GE, FTGE, RATIO, PEAK, CENTRAL = load_pupil_rankings(path)
    heuristic_rankings = [RWGE, FTRWGE, GE, FTGE, RATIO, PEAK, CENTRAL]
    heuristic_names = ["RWGE", "FTRWGE", "GE", "FTGE", "RATIO", "PEAK", "CENTRAL"]
    
    # Get list of all pupil ids
    pupil_ids = [entry[1] for entry in rankings]
    pupil_ranking_values = [entry[2] for entry in rankings]
        
    value = get_ranking_value(pupil_object, rankings, heuristic_rankings)
                
    # Check that the new pupil is in the top 25
    if value >= pupil_ranking_values[-1]:
        rmtree("{}{}".format(path, pupil_object.uid))
        return False # Exit out without changing anything
        
    # Insert into absolute rankings
    for i in range(len(pupil_ranking_values)):
        if value < pupil_ranking_values[i]:
            index = i
            break
            
    # Split list about insertion index 
    first = rankings[:index]
    second = rankings[index:-1]
    
    # Increase ranking of each entry in second
    for entry in second:
        entry[0] += 1
    
    # Get id if pupil to delete
    to_delete = rankings[-1][1]
    
    # Replace worst pupil id with new pupil id
    replace_index = pupil_ids.index(to_delete)
    pupil_ids[replace_index] = int(pupil_object.uid)
    
    # Create new rankings list   
    new_entry = [index, pupil_object.uid, value]
    new_rankings = first + [new_entry] + second
    
    # Delete new worst pupil from all lists
    for i in range(len(heuristic_rankings)):
        rankings_list = heuristic_rankings[i]
        
        # Loop over each element in the heuristic text files
        for j in range(len(rankings_list)):
            if rankings_list[j][1] == to_delete:
                pupil_entry = [-1, int(pupil_object.uid), pupil_object.heuristics[heuristic_names[i]]]
                rankings_list[j] = pupil_entry
                
        # Re sort list by heuristic values
        if heuristic_names[i] == "PEAK" or heuristic_names[i] == "CENTRAL":
            rankings_list.sort(key=lambda x: x[2])
        else:
            rankings_list.sort(key=lambda x: x[2], reverse=True)
        
        # Re-assing rankings to each
        for j in range(len(rankings_list)):
            rankings_list[j][0] = j
    
    # Re-evalute the absolute rankings
    final_ranking_values = list(np.zeros(len(pupil_ids)))
            
    # Iterate over each heursitic
    for rankings_list in heuristic_rankings:
        rankings_list.sort(key=lambda x: x[2]) # sort by value
        for i in range(len(rankings_list)):
            rank = rankings_list[i][0]
            entry_id = rankings_list[i][1]
            index = pupil_ids.index(entry_id)
            final_ranking_values[index] += rank/len(heuristic_rankings)

    # Create the new pupil object entries
    pupil_values_ids = [[final_ranking_values[i], pupil_ids[i]] for i in range(len(pupil_ids))] 
    pupil_values_ids.sort(key=lambda x: x[0])
    
    new_rankings = []
    for i in range(len(pupil_values_ids)):
        new_rankings.append(str([i, pupil_values_ids[i][1], pupil_values_ids[i][0]]))
    
    # Invert order of lists (side effect of using .sort())
    for i in range(len(heuristic_rankings)):
        rankings_list = heuristic_rankings[i]
        if not(heuristic_names[i] == "PEAK" or heuristic_names[i] == "CENTRAL"):
            rankings_list.reverse()

    # Delete current worst pupil
    rmtree("{}{}".format(path, to_delete))
        
    # Save new pupil ordering to files
    heuristic_entries = [new_rankings, RWGE, FTRWGE, GE, FTGE, RATIO, PEAK, CENTRAL]
    save_pupil_rankings(path, heuristic_entries)
    
    return True

def get_ranking_value(pupil_object, rankings, heuristic_rankings):
    """
    Gets the relative ranking of the pupil_object to the rest of the pupils in current circulation
    """
    # Get the values from the pupil
    rwge_pupil = pupil_object.heuristics["RWGE"]
    ftrwge_pupil = pupil_object.heuristics["FTRWGE"]
    ge_pupil = pupil_object.heuristics["GE"]
    ftge_pupil = pupil_object.heuristics["FTGE"]
    ratio_pupil = pupil_object.heuristics["RATIO"]
    peak_pupil = pupil_object.heuristics["PEAK"]
    central_pupil = pupil_object.heuristics["CENTRAL"]
        
    # Get relative rankings of each heuristic
    ranks = {"RWGE": 25, "FTRWGE": 25, "GE": 25, "FTGE": 25, "RATIO": 25, "PEAK": 25, "CENTRAL": 25}
    for i in reversed(range(len(rankings))):
        # Maximisation Heuristics
        if rwge_pupil > heuristic_rankings[0][i][2]:
            ranks["RWGE"] = i
        if ftrwge_pupil > heuristic_rankings[1][i][2]:
            ranks["FTRWGE"] = i
        if ge_pupil > heuristic_rankings[2][i][2]:
            ranks["GE"] = i
        if ftge_pupil > heuristic_rankings[3][i][2]:
            ranks["FTGE"] = i
        if ratio_pupil > heuristic_rankings[4][i][2]:
            ranks["RATIO"] = i
        # Minimisation Heuristics
        if peak_pupil < heuristic_rankings[5][i][2]:
            ranks["PEAK"] = i
        if central_pupil < heuristic_rankings[6][i][2]:
            ranks["CENTRAL"] = i
            
    # Set relative weights
    weights = {}
    weights["RWGE"] = 0
    weights["FTRWGE"] = 1
    weights["GE"] = 0
    weights["FTGE"] = 0
    weights["RATIO"] = 1
    weights["PEAK"] = 2
    weights["CENTRAL"] = 0
                        
    # Calculate average (absolute) ranking
    value = 0
    for key in ranks.keys():
        # We want to value the peak pixel value greater than the others
        if weights[key] != 0:
            value += ranks[key]/weights[key]
            
    return value

def load_pupil_rankings(path):
    """
    Loads the pupil rankings from the text files to a dictionary of lists for evalution
    """
    file_names = ["rankings", "RWGE", "FTRWGE", "GE", "FTGE", "RATIO", "PEAK", "CENTRAL"]
    
    lists_out = []
    for file_name in file_names:
        values = []
        with open("{}/{}.txt".format(path, file_name), 'r') as f:
            data = f.readlines()
            for line in data:
                if line == "\n":
                    continue
                    
                rank, unique_id, value = line.strip()[1:-1].split(", ")
                rank = int(rank)
                unique_id = int(unique_id)
                value = float(value)
                values.append([rank, unique_id, value])
        lists_out.append(values)
        
    return lists_out[0], lists_out[1], lists_out[2], lists_out[3], lists_out[4], lists_out[5], lists_out[6], lists_out[7]

def save_pupil_rankings(path, entries):
    """
    saves the pupil rankings from a list of lists
    """
    file_names = ["rankings", "RWGE", "FTRWGE", "GE", "FTGE", "RATIO", "PEAK", "CENTRAL"]
        
    for file_name, pupil_rankings in zip(file_names, entries):
        values = []
        with open("{}/{}.txt".format(path, file_name), 'w') as f:
            for entry in pupil_rankings:
                f.write("{}\n".format(str(entry)))
                