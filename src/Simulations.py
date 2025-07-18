from src.Data import * # Class containing the data
from src.Assignment import * # Class containing an assignment
from src.Model import * # Class containing a Pulp model used for optimization
from src.ModelColumnGen import * # Class containing Pulp model that optimizes using column generation
from src.ModelFracStable import * # Class containing a Pulp model for finding an fractionally stable stochastic improvement
from src.ModelHeuristicLP import * # Class containing heuristic that will use generated weakly stable matchings as an input
from src.DataGen import * # Generate student preferences and school priorities
from src.DataGenEE import * # Generate data according to the method by Erdil & Ergin (2008)
from src.DA_STB import * # Generate DA assignment with single tie-breaking (STB)
from src.ErdilErgin import * # Erdil & Ergil their implementation of Stable Improvement Cycles algorithm + alternative implementation DA
from src.SICs import * # Adaptation of SICs algorithm to our code

import random
import pickle # to export data


def SimulationCG(n_students_schools: list, alpha: list, beta: list, n_iterations_simul: int, n_match: int, time_lim: int, seed: int, n_sol_pricing: int, gap_pricing: float, MIPGap: float, bool_ColumnGen: bool, print_out = False):
    """
    Will run column generation framework 'n_iterations' times, for the specified parameter values
    Output: an array containing SolutionReport objects

    - n_students_schools, alpha, beta: lists containing the corresponding parameter values
    - n_students_schools is a list of pairs of values to evaluate: e,g,. [[100, 2], [200, 4], [400, 8]]
    - n_sol_pricing: number of solutions returned by pricing problem
    - gap_pricing: optimality gap used for the solutions included in the solution pool in the pricing problem
    - bool_ColumnGen: if True: perform entire column generation for time_limit period
                      if False: only perform first iteration, and don't build pricing problem

    For each combination of parameter values (n_stud, n_school, alpha, beta), we start seed from same value
    This makes it easier to reproduce the results later on
    """
    print_intermediate = True

    # Create directory if it doesn't exist
    os.makedirs('Simulation Results', exist_ok=True)

    S_vector = []

    now = time.strftime('%Y-%m-%d_%H%M%S')


    total_combinations = len(n_students_schools) * len(alpha) * len(beta)

    for k, b, a in tqdm(itertools.product(n_students_schools, beta, alpha), total = total_combinations, desc='Data instances', unit = 'inst', disable= not print_out):
        # Generate data using data generation by Erdil & Ergin (2008)
        random.seed(seed)

        n = k[0] # Number of students
        m = k[1] # Number of schools

        seed_vector = []
        for i in range(n_iterations_simul):
            seed_vector.append(random.randint(0,1000000000))

        for i in tqdm(range(n_iterations_simul), desc = 'iterations', total = n_iterations_simul, unit = 'iter', disable = not print_out):
            if print_out:
                print('\nn,m,alpha, beta, seed', n, m, a, b, seed_vector[i])
            
            pref_list_length = m # Assume pref_list_length = number of schools (as they do)
            MyData = DataGenEE(n, m, a, b, pref_list_length, False, seed_vector[i])

            # Generate the assignment from DA with Single Tie-Breaking with n_match samples
            A = DA_STB(MyData, n_match, 'GS', False, 0, print_intermediate)

            # Find Stable improvement cycles à la Erdil and Ergin (2008)
            A_SIC = SIC_all_matchings(MyData, A, print_intermediate)

            # Solve the formulations
            MyModel = ModelColumnGen(MyData, A_SIC, A.assignment, print_intermediate)
                # Will use matchings in A_SIC to sd_dominate the assignment 'A.assignment' (found by DA)
            
            S = MyModel.Solve("TRAD", "GUROBI", print_log=False, time_limit= time_lim, n_sol_pricing= n_sol_pricing, gap_solutionpool_pricing=gap_pricing, MIPGap=MIPGap, bool_ColumnGen=bool_ColumnGen, print_out=print_intermediate)

            S.Data = copy.deepcopy(MyData)
            S.n_stud = n
            S.n_schools = m
            S.alpha = a
            S.beta = b
            S.seed = seed_vector[i]

            S_vector.append(S)

            # Save results of simulations using pickle
            name = 'Simulation Results/SIM_' + now + '.plk'
            
            # Save to file
            #with open(name, 'wb') as f: # Rewrites the whole S_vector to the output file
            #    pickle.dump(S_vector, f)
            with open(name, 'ab') as f: # Appends just 'S'
                pickle.dump(S, f)


    return S_vector