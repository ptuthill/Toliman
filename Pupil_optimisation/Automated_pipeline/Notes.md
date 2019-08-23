# Pupil pipeline 

## To Do:
- Vectorise all the fucntions!
    - Big tasks
    - Requires lots of testing
    - Ensure to doccument carefully along the way
    
- Use TensorFlow to optimise specific heuristics
    - Big task (requires learning TensorFlow which is probably a good thing anyway)
    - Need to discuss with Kieran what are the most optimal heristics to optimise
    - Possibly find maxiamlly optimised pupils for each heurstic (is this valuable?)
    
- Heuristic aditions:
    - Variance of local maximums
    - Average distance of local maximums
    - Number of maximums in some range? Max peak of 20 local maximums?

___

## Known Bugs:
- Ranking data is stored/read from the wrong file. (easy fix, potential restructure of the way data is stored)
    - Fix by taking a dictionary based approach rather than assuming an order
    - Potentially create a batch class to store the data? *Big task*
        - Easier interaction (no need to work with formatting)
        - Also allows for easier ranking of pupils
        - Perhaps remove file storage all together or only store the data from every ith teration
    > Heuristic -> (stored in) File name
    > Peak -> ratio
    > central -> peak
    > ratio -> RWGE
    > GE -> Central
    > FTRWGE -> FTGE
    > FTGE -> GE 
    > RWGE -> FTRWGE
    
- Load_pupil creates a new UUID for the pupil object it returns
    - Likely due to the fact that it recreates the pupil from the data rather than loading the actual pupil file
    - Potentially fixed with creating a batch class to store the data
    - Else trace the functions called by it to see which creates the new UUID
        - Could be from checking that a pupil with that UUID already exists (fix with batch class!)
        
- G-Saxburg WF modifier seems to be behaving strangely
    - Needs to be tested to find cause of issue
    