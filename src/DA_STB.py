from .Assignment import *
from .GaleShapley import gale_shapley

from numpy.random import default_rng
import time
from tqdm import tqdm # To show progress bar
import math
import itertools

def DA_STB(MyData: Data, n_iter: int, seed = 123456789, print_out = False):
    """
    Deferred Acceptance with single tie-breaking

    Parameters:
    - MyData: An instance from the Data class
    - n_iter: number of tie-breakings sampled
    - print_out: boolean to control output on the screen

    Returns:
    - An instance of the Assignment class
    """

    if seed != 123456789:
        # Use seed in argument
        rng = default_rng(seed)
    else:
        # Generate random seed 
        # Create a seed based on the current time
        seed = int(time.time() * 1000) % (2**32)  # Modulo 2^32 to ensure it's a valid seed

    np.random.seed(seed)

    # First, check how many tie-breaking rules would be needed in total
    # Look at total number of students who are included in ties
    students_in_ties = set()
    for j in range(MyData.n_schools):
        for k in range(len(MyData.prior[j])):
            if len(MyData.prior[j][k]) >= 2: # When more than a single student in this element
                for l in range(len(MyData.prior[j][k])):
                    # students_in_ties.add(MyData.ID_stud.index(MyData.prior[j][k][l])) # We add the index of this student, not its name
                    students_in_ties.add(MyData.prior[j][k][l])

    students_in_ties = list(students_in_ties) # Convert the set to a list, allows us to access k-th element
    
    # The total number of needed tie-breaking rules is m!, where m = |student_in_ties|
    n_STB = math.factorial(len(students_in_ties))

    # We only need to perturb the students who appear in ties:
    
    if n_STB < n_iter:
        n_iter = n_STB
        # Enumerate all relevant permutations
        permut = list(itertools.permutations(students_in_ties))
    else:
        permut = set() # We first create a set, to ensure that all found permutations are unique. Later, convert to list
        # Sample n_iter out of all n_STB relevant permutations
        while len(permut) < n_iter:
            np.random.shuffle(students_in_ties)  # Shuffle in place
            permut.add(tuple(students_in_ties))
        permut = list(permut)
    
    if print_out:
        print(f"Students in ties: {len(students_in_ties)}")
        print(f"Tie-breaking rules needed: {n_STB}")
        print(f"Tie-breaking rules sampled: {n_iter}")
        # print(f"permut: {permut}")

    # For each of the permutations, break ties in the preferences and run Gale-Shapley algorithm on them
    M_sum = np.zeros(shape=(MyData.n_stud, MyData.n_schools)) # Will contain the final random_assignment

    for p in tqdm(permut):
        prior_new = [] 
        for j in range(len(MyData.prior)):
            # Just add priorities if no ties:
            if len(MyData.prior[j]) == MyData.n_stud:
                prior_new.append(MyData.prior[j])
            else:
                prior_array = []
                for k in range(len(MyData.prior[j])):
                    if len(MyData.prior[j][k]) == 1:
                        prior_array.append(MyData.prior[j][k])
                    else: # set of students who have same priorities
                        # Reorder the students based on the permuation
                        reordered_prior = list(sorted(MyData.prior[j][k], key=lambda x: p.index(x)))

                        # Add to prior_array
                        for l in range(len(MyData.prior[j][k])):
                            prior_array.append(reordered_prior[l])
                prior_new.append(prior_array)
                
        # Compute DA matching for the new priorities after tie-breaking
        Data_new_prior = Data(MyData.n_stud, MyData.n_schools, MyData.pref, prior_new, MyData.cap, MyData.ID_stud, MyData.ID_school, MyData.file_name)
        M_sum = M_sum + gale_shapley(Data_new_prior)            
        
    M_sum = M_sum / n_iter

    # Create an instance of the Assignment class
    label = MyData.file_name + "_" + "DA_STB" + str(n_iter)
    A = Assignment(MyData, M_sum, label)

    return A

    