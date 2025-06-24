from .Assignment import *
# Install necessary packages before running the script:
# pip install pulp
# pip install gurobipy
# pip install pyscipopt
# Note: Obtain an academic license for Gurobi from: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

# Check with solvers available on computer
import pulp as pl
from pulp import *

import time

import gurobipy

from contextlib import redirect_stdout

# To check which solvers available on computer:
# print(pl.listSolvers(onlyAvailable=True))

class SolutionReport:
    """
    Will be output by Solve function. Will be saved using pickle, and analyzed afterwards
    """
    # Information added to solution report during data generation
    Data: Data          # Contains data instance
    n_stud: int
    n_schools: int
    alpha: float        # alpha and beta will only be filled in when using data generation Erdil & Ergin
    beta: float
    seed: int

    # Information added after solution is found
    A: Assignment       # Contains the final assignment
    A_SIC: Assignment   # Contains warm start solution (in general, SICs by Erdil & Ergin)
    A_DA_prob: np.ndarray # Assignment probabilities to sd-dominate (in general, DA)
    avg_ranks: dict     # Contains average ranks of several solutions along the process
    obj_master: list    # Objective values of master in iterations
    obj_pricing: list   # Objective values of pricing in iterations
    n_iter: int         # Number of iterations
    time_limit_exceeded: bool # Whether time limit is exceeded
    time_limit: int     # Time limit
    optimal: bool       # Optimality guaranteed?
    time: float         # Time used
    Xdecomp: list       # Matchings in the found decomposition
    Xdecomp_coeff: list # Weights of these matchings
    #... 





class ModelColumnGen: 
    """
    Contains two methods:
        __init__: initializes the model, and the solver environment

        Solve: solves the model.
            The parameters of this method can control which objective function is optimized, and which solver is used
    """
    
    # Used this example as a template for Pulp: https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
    
    def __init__(self, MyData: Data, p: Assignment, p_DA: np.ndarray, print_out: bool):
        """
        Initialize an instance of Model.

        Args:
            MyData (type: Data): instance of class Data.
            p (type: Assignment): instance of class Assignment (possibly including SICs). This will be used for warm start solution
            p_DA (type: np.ndarray): this is the probabilistic assignment which we want to sd_dominate
            print_out (type: bool): boolean that controls which output is printed.
            nr_matchings (optional): number of matchings used in the decomposition, optional parameter that defaults to n_students * n_schools + 1

        """
        self.MyData = copy.deepcopy(MyData)
        self.p = copy.deepcopy(p)
        self.p_DA = copy.deepcopy(p_DA)

        # Create the pulp model
        # 'self.master' refers to master problem
        # 'self.pricing' will refer to pricing problem
        self.master = LpProblem("Improving_ex_post_stable_matchings", LpMinimize)

        # Create variables to store the solution in
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings
        zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
        self.Xassignment = Assignment(MyData, zero) # Contains the final assignment found by the model

        # Ranges that will help with coding
        self.STUD = range(0,self.MyData.n_stud)
        self.SCHOOLS = range(0, self.MyData.n_schools)

        # Tuple with all student-school pairs that are preferred to outside option
        # This tuple contains the INDICES of the students and the pairs, and not their original names
        self.PAIRS = []
        for i in range(0, MyData.n_stud):
            for j in range(0,len(MyData.pref[i])):
                # Convert pref[i][k] (school ID as string) to column index
                col_index = MyData.ID_school.index(MyData.pref[i][j])
                self.PAIRS.append((i,col_index))   
        
        # First, convert the set containing all matchings to a list to make it subscribable
        self.M_list = list(p.M_set)

        # M[k][i][j] = 1 if student i is assigned to school j in matching k, and 0 otherwise
        self.nr_matchings = len(self.M_list)
        self.N_MATCH = range(self.nr_matchings)  # Number of matchings

        # Create the parameter set M[k][i][j]
        self.M = np.zeros((self.nr_matchings, self.MyData.n_stud, self.MyData.n_schools))
        for k in self.N_MATCH:
            for i in range(self.MyData.n_stud):
                for j in range(self.MyData.n_schools):
                    self.M[k, i, j] =self.M_list[k][i][j]  # Fill the parameter from the M_list

        #self.M = LpVariable.dicts("M", [(k, i, j) for k in self.N_MATCH for i, j in self.PAIRS], cat="Binary")

        # Store labels to make understanding output easier
        self.labels = {}
        for k in self.N_MATCH:
            for i in range(self.MyData.n_stud):
                for j in range(self.MyData.n_schools):
                    student_name = self.MyData.ID_stud[i]
                    school_name = self.MyData.ID_school[j]
                    self.labels[k, i, j] = f"M_{k}_{student_name}_{school_name}"



        #### OBJECTIVE FUNCTION ####
        # Add an empty objective function
            # Every time you update it, you should add it to the model again
                # using a code like:
                # self.model.setObjective(self.model.objective+obj_coeff*self.w[m])
        self.master += LpAffineExpression()
            
            
            
            
        #### CONSTRAINTS ####
        # Other constraints defined for specific models in functions below (see function Solve)
        
        self.constraints = {}

        # Ensure weights sum up to one
        # We save this constraint in order to later add decision variables to it.
        self.constraints["Sum_to_one"] = LpConstraintVar("Sum_to_one", LpConstraintEQ, 1)

        # Add one constraint for each pair to model first-order stochastic dominance
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                school_name = self.MyData.pref_index[i][j]
                # Compute original probability of being assigned to j-th ranked school or better
                original_p = 0
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    original_p += self.p_DA[i,pref_school]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                self.constraints[name]=LpConstraintVar(name, LpConstraintGE, original_p)

        # Non-negativity (explicitly included to get dual variables)
        #for m in self.N_MATCH:
        #    name = 'GE0_' + str(m)
        #    self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
        
        # Add all these contraints to the model
        for c in self.constraints.values():
            self.master += c

        

        #### DECISION VARIABLES ####
        self.w = []
        #print(self.nr_matchings)
        for m in tqdm(self.N_MATCH, desc='Master: add decision variables', unit='var', disable= not print_out):
            #print(self.M[m])
            self.add_matching(self.M[m], m, print_out)



        # Create variables, and add them to the constraints
        
        # w[k] is the weight of matching k in the decomposition
        #self.w = LpVariable.dicts("w", self.N_MATCH, 0, 1,  )
        #self.w = LpVariable("w", self.nr_matchings, lowBound=0, upBound=1, e={self.constraints["Sum_to_one"]:1} )

        #self.w = LpVariable("w", self.nr_matchings, e={self.constraints["Sum_to_one"]:1} )
        #self.master += 2*self.w

        
        #self.vars += self.w()
        #for m in self.N_MATCH:
        #    for c in range(1,len(self.constraints)):
        #        self.master.addVariableToConstraints(self.w[m], {self.constraints[c], 1})

        #self.master.writeLP("TestColumnFormulation.lp")

        
        # Set the warm start solution as the decomposition found after SICs
        for m in self.N_MATCH:
            # Find matching (because w_set is a set and not subscriptable)
            M = self.M_list[m]
            self.w[m].setInitialValue(self.p.w_set[M])
        
        
        
      
    def add_matching(self, M_in: np.ndarray, index, print_out: bool):
        """
        Function to add a matching M as a decision variable to the master proble
        Index is the index of the matching in the master problem
        """  
        
        # First, determine coefficients of this variable in the constraints
        coeff = {}
        coeff["Sum_to_one"] = 1
        coeff["Obj"] = 0 # Objective coefficients will be fixed later
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                school_name = self.MyData.pref_index[i][j]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                
                # Initialize the coefficient if it doesn't exist
                # (needed because we use +=, and not =)
                if name not in coeff:
                    coeff[name] = 0
                
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    #print(m,i,j,pref_school, self.M[m,i,pref_school])
                    coeff[name] += M_in[i,pref_school]

        # Non-negativity of the variables
        # First, add a new constraint, and only then set the coefficient
        #for m in self.N_MATCH: 
        #    name = 'GE0_' + str(m)
        #    coeff[name] = 0
        #name = 'GE0_' + str(index)
        #coeff[name] = 1
        
        # Then, create a dictionary for `e` that maps constraints to their coefficients
        e_dict = {self.constraints[key]: coeff[key] for key in self.constraints if coeff[key] > 0}

        # Add this variable to self.w
        name_w = "w_" + str(index)
        self.w.append(LpVariable(name_w, lowBound= 0, e=e_dict))
        
                        
        # Compute objective coefficient of this variable (average rank)
        obj_coeff = 0
        for (i,j) in self.PAIRS:
            obj_coeff += M_in[i,j]*(self.MyData.rank_pref[i,j]+ 1) / self.MyData.n_stud # + 1 because the indexing starts from zero

        # Add this variable to the model with the correct objective coefficient
        self.master.setObjective(self.master.objective+obj_coeff*self.w[index])

        #self.master.writeLP("TestColumnFormulation.lp")




        
    def Solve(self, stab_constr: str, solver: str, print_log: str, time_limit: int, print_out: bool):
        """
        Solves the formulation using column generation.
        Returns an instance from the Assignment class.

        Note that, if you create an object of ModelColumnGen using an assignment object containing SICs already,
        then those will be the matchings that are used.

        Args:
            stab_constr (str): controls which type of stability constraints are used.
            solver (str): controls which solver is used. See options through following commands:
                solver_list = pl.listSolvers(onlyAvailable=True)
                print(solver_list)
            print_out (bool): boolean that controls which output is printed.
        """
        # Compute average rank of current assignment
        self.avg_rank_DA = 0
        for (i,j) in self.PAIRS:
            self.avg_rank_DA += self.p_DA[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
        # Average
        self.avg_rank_DA = self.avg_rank_DA/self.MyData.n_stud
        #if print_out == True:  

        self.avg_rank = 0
        for (i,j) in self.PAIRS:
            self.avg_rank += self.p.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
        # Average
        self.avg_rank = self.avg_rank/self.MyData.n_stud
        if print_out:   
            print(f"\nAverage rank DA : {self.avg_rank_DA}.\n")
            print(f"\nAverage rank warm start solution : {self.avg_rank}.\n\n")
        
        
        # Check that strings-arguments are valid

        # Valid values for 'solver'
        solver_list = pl.listSolvers(onlyAvailable=True)
        if solver not in solver_list:
           raise ValueError(f"Invalid value: '{solver}'. Allowed values are: {solver_list}")

        # Valid values for 'stab_constr'
        stab_list = ["TRAD"]
        if stab_constr not in stab_list:
           raise ValueError(f"Invalid value: '{stab_constr}'. Allowed values are: {stab_list}")        

        self.time_limit_exceeded = False
        self.time_limit = time_limit

        #### PRICING PROBLEM ####
        # Defined in separate function
        self.build_pricing(stab_constr, print_out)

       
        
        #### RUN COLUMN GENERATION PROCEDURE ####
        optimal = False
        
        #self.pricing.writeLP("PricingProblem.lp")
        self.iterations = 1
        if print_out:
            print("Number of matchings:", self.nr_matchings)

        # Create two empty arrays to store objective values of master and pricing problem
        self.obj_master = []
        self.obj_pricing = []

        starting_time = time.monotonic()

        while (optimal == False):
            if print_out:
                print('ITERATION:', self.iterations)            
            if print_out:
                if print_out:
                    print("\n ****** MASTER ****** \n")
                #for m in self.N_MATCH:
                #    print(self.M_list[m])

            # String can't be used as the argument in solve method, so convert it like this:
            solver_function = globals()[solver]  # Retrieves the GUROBI function or class
        
            # Solve the formulation
            if print_log == False:
                self.master.solve(solver_function(msg = False, logPath = "Logfile_master.log", warmStart = True))
            else:
                self.master.solve(solver_function(msg = True, warmStart = True))
            #self.model.solve(GUROBI_CMD(keepFiles=True, msg=True, options=[("IISFind", 1)]))
            
            # Get objective value master problem
            self.obj_master.append(self.master.objective.value())
            if print_out:
                print("Objective master: ", self.obj_master[-1])

            # Get average rank of first iteration
            if self.iterations == 1:
                self.avg_rank_first_iter = self.obj_master[-1]

            # Check if the time limit is exceeded
            current_time = time.monotonic()
            if current_time - starting_time > time_limit:
                optimal = True
                self.time_limit_exceeded = True
                return self.generate_solution_report(print_out)

            #if print_out:
            #    for m in self.N_MATCH:
            #        print("w_", m, self.w[m].value())
            
            #### SOLVE PRICING ####
            # Get dual variables
            duals = {}
            
            duals["Sum_to_one"] = self.master.constraints["Sum_to_one"].pi

            for i in self.STUD:
                for j in range(len(self.MyData.pref[i])):
                    school_name = self.MyData.pref_index[i][j]
                    name_duals = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                    name_constr = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                        
                    duals[name_duals]=self.master.constraints[name_constr].pi

            #for m in self.N_MATCH:
            #    name_GE = 'GE0_' + str(m)
            #    duals[name_GE] = self.master.constraints[name_GE].pi
            #if print_out:
            #    print(duals)
            
            #for name in duals:
            #    if duals[name] > 0:
            #        print(name, duals[name])
                
            # Modify objective function pricing problem
            # Careful, don't add constant terms, won't be taken into consideration!
            pricing_obj = LpAffineExpression()
            for i in self.STUD:
                #print('student ', i)
                for j in range(len(self.MyData.pref[i])): 
                    school_name = self.MyData.pref_index[i][j]
                    pricing_obj -= self.M_pricing[i,school_name] * (self.MyData.rank_pref[i,school_name]+ 1) / self.MyData.n_stud # + 1 because the indexing starts from zero
                    #print('  school ', school_name, -(self.MyData.rank_pref[i,school_name]+ 1) / self.MyData.n_stud)
                    for k in range(j+1):
                        pref_school = self.MyData.pref_index[i][k]
                        name = "Mu_" +  str(self.MyData.ID_stud[i]) + "_" + str(pref_school)
                        pricing_obj += self.M_pricing[i,pref_school] * duals[name]
                        #print('      school ', pref_school, duals[name])

            #if print_out:
            #    print(pricing_obj)
            #print("Duals Sum_to_one", duals["Sum_to_one"])
            #print("Duals", duals)


            self.pricing.setObjective(pricing_obj)


            
            #self.pricing.writeLP("PricingProblem.lp")

            # Add the constant terms!
            constant = 0
            constant += duals['Sum_to_one']
                # This constant term will not be printed in the objective function in the .lp file, I think

            #for m in self.N_MATCH:
            #    name_GE = 'GE0_' + str(m)
                #constant += duals[name_GE]
            if print_out:
                print("Constant term", constant)
            
            # Solve modified pricing problem
            if print_out:
                print("\n ****** PRICING ****** \n")
            
            constant_str = str(constant)

            # Add warm start to pricing problem by referring to previously found solution
            #for (i,j) in self.PAIRS:
            #    self.M_pricing[(i,j)].setInitialValue(self.M_list[-1][i][j])
            
            # Update time limit:
            current_time = time.monotonic()
            new_time_limit = max(time_limit - (current_time - starting_time), 0)
            if print_out:
                print('New time limit', new_time_limit)

            if print_log == True:  
                #self.pricing.solve(solver_function())
                
                #self.pricing.solve(solver_function(timeLimit = new_time_limit, BestObjStop = -constant +0.0001))
                self.pricing.solve(solver_function(timeLimit = new_time_limit, MIPGap = 0.1))
                # Will stop the solver once a matching with objective function at least zero has been found
                #self.pricing.solve(solver_function())

            else:
                #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull):
                        #self.pricing.solve(solver_function(msg=False, timeLimit=new_time_limit, logPath = 'Logfile_pricing.log',BestObjStop = -constant + 0.0001))
                        self.pricing.solve(solver_function(msg=False, timeLimit=new_time_limit, logPath = 'Logfile_pricing.log',MIPGap = 0.10))
                #self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))
            

            # If not infeasible:

            
            if self.pricing.status not in [0,-1]:
                #print("Status code: ", self.pricing.status)
                obj_pricing_var = self.pricing.objective.value() + constant
                self.obj_pricing.append(obj_pricing_var)
                if print_out:
                    print("\t\tObjective pricing: ", obj_pricing_var)

            #if print_out:
            if False:
                for (i,j) in self.PAIRS:
                    print("M[",i,j,'] =', self.M_pricing[i,j].value())
            
            #### EVALUATE SOLUTION ####
            if print_out:
                print('Pricing status', self.pricing.status)
            if self.pricing.status == 0: # Time limit exceeded
                self.time_limit_exceeded = True
                optimal = False
                self.time_columnGen = self.time_limit
                return self.generate_solution_report(print_out) 
            
            elif self.pricing.status != -1:            
                if obj_pricing_var > 0:
                    # The solution of the master problem is not optimal over all weakly stable matchings
                    
                    # Add non-negativity constraint to the master for this new matching
                    #name = 'GE0_' + str(len(self.w))
                    #self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
                    #self.master += self.constraints[name]
                    
                    # Add the matching found by the pricing problem to the master problem       
                    found_M = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
                    for (i,j) in self.PAIRS:
                        found_M[i][j] = self.M_pricing[i,j].value()
                    #print('Found_M', found_M)
                    
                    self.add_matching(found_M, len(self.w), print_out)
                    
                    self.M_list.append(found_M)
                    self.nr_matchings += 1
                    if print_out:
                        print("New number of matchings:", self.nr_matchings)

                    self.N_MATCH = range(self.nr_matchings)

                    # Exclude this matching from being find by the pricing problem in the future.
                    self.pricing += lpSum([self.M_pricing[i,j] * found_M[i][j] for (i,j) in self.PAIRS]) <= lpSum([found_M[i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{self.nr_matchings-1}"
                    
                    #print("Matching added.")
                    #if print_out:
                    #    print(found_M)

                    #if self.iterations == 10:
                    #    optimal = True
                    #    print("Process terminated after ", self.iterations, " iterations.")
                    self.iterations = self.iterations + 1                

                    
                else:
                    optimal = True  
                    current_time = time.monotonic()
                    self.time_columnGen = current_time - starting_time
                    return self.generate_solution_report(print_out)    
            else:
                optimal = True
                current_time = time.monotonic()
                self.time_columnGen = current_time - starting_time
                return self.generate_solution_report(print_out)     


            

    def build_pricing(self, stab_constr: str, print_out: bool):
        # Create Pulp model for pricing problem
        self.pricing = LpProblem("Pricing problem", LpMaximize)

        # Decision variables
        self.M_pricing = LpVariable.dicts("M", [(i, j) for i, j in self.PAIRS], cat="Binary")
        # Rename M
        for i, j in self.M_pricing:
            student_name = self.MyData.ID_stud[i]
            school_name = self.MyData.ID_school[j]
            self.M_pricing[i, j].name = f"M_{student_name}_{school_name}"

        ### CONSTRAINTS ###

        if stab_constr == 'TRAD':
            # Stability
            for i in self.STUD:
                for j in range(len(self.MyData.pref_index[i])):
                    current_school = self.MyData.pref_index[i][j]
                    lin = LpAffineExpression()

                    lin += self.MyData.cap[current_school] * self.M_pricing[i, current_school]

                    # Add all schools that are at least as preferred as the j-ranked school by student i
                    for l in range(j):
                        lin += self.MyData.cap[current_school] * self.M_pricing[i,self.MyData.pref_index[i][l]]


                    # Add terms based on priorities
                    prior_current = self.MyData.rank_prior[current_school][i]
                    for s in self.STUD:
                        if s != i:
                            # If current_school ranks student s higher than student i
                            if self.MyData.rank_prior[current_school][s] <= self.MyData.rank_prior[current_school][i]:
                                if (s, current_school) in self.PAIRS:
                                    lin += self.M_pricing[s,current_school]

                    # Add to model:
                    name = "STAB_" + str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_school[current_school]) 
                    self.pricing += (lin >= self.MyData.cap[current_school], name) 

        elif stab_constr == "CUTOFF":
            # Create decision variables cutoff scores, one for each school
            # NOTICE: The score of a student for a school, is the rank of the priority group to which they belong on that school
            # Contrary to the literature, all students with a LOWER score than the cutoff score will be admitted

            # NOTICE as well: We slightly modify non-envy from paper Agoston et al. (2022)
                # By removing the epsilon in constraints (6), we allow that some students from a priority group
                # are assigned, while others from the same priority group are not assigned because of capacities
                # This is the Irish system, and not the Hungarian or the Chilean (as they are called in that paper)
            
            self.t = LpVariable.dicts("t", [j for j in self.SCHOOLS], cat="Continuous")

            # Auxiliary parameter that contains number of priority groups at each school
            s_max = []
            for j in self.SCHOOLS:
                s_max.append(len(self.MyData.prior[j]))
            

            for (i,j) in self.PAIRS:
                # Find priority group to which i belongs at school j
                s_i_j = s_max[j]
                for k in range(s_max[j]):
                    if isinstance(self.MyData.prior[j][k], tuple): # When more than a single student in this element
                        if i in self.MyData.prior_index[j][k]:
                            s_i_j = k

                self.pricing += ((1 - self.M_pricing[i][j]) * s_max[j] + s_i_j <= self.t[j])

                #######################################################
                #################### NOT FINISHED YET #################
                #### How can some students in a priority be assigned ##
                ## And some others not, using cutoff scores? ##########
                #######################################################

        
        # Each student at most assigned to one school
        for i in self.STUD:
            self.pricing += lpSum([self.M_pricing[i,j] for j in self.SCHOOLS if (i,j) in self.PAIRS]) <= 1, f"LESS_ONE_{l}_{i}"

        # Capacities schools respected
        for j in self.SCHOOLS:
            self.pricing += lpSum([self.M_pricing[i,j] for i in self.STUD if (i,j) in self.PAIRS]) <= self.MyData.cap[j], f"LESS_CAP_{l}_{j}"
         
        # Exclude matchings that are already found:
        # Simple "no-good" cuts, where you sum matched student-school pairs for matching l, and force the sum to be strictly smaller
        # Required, because many matchings have same objective value in pricing problem,
            # and, sometimes, when a matching is added to the master and not immediately used, 
            # the dual prices are the same or similar, and the matching could have been found again by the pricing problem
        for l in tqdm(self.N_MATCH, desc='Pricing exclude found matchings', unit='matchings', disable=not print_out):            
            self.pricing += lpSum([self.M_pricing[i,j] * self.M_list[l][i][j] for (i,j) in self.PAIRS]) <= lpSum([self.M_list[l][i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{l}"
        

    def pricing_opt_solution(self, avg_rank_DA, avg_rank, print_out: str):
        
        if self.time_limit_exceeded == False:  
            # Optimal solution of master problem!
            if print_out:
                print("Optimal solution found!\nBest average rank: ", self.obj_master[-1])

        else:
            # Time limit exceeded
            if print_out:
                print('\nTime limit of ', self.time_limit, "seconds exceeded!\n")
                print('Rank best found solution:', self.obj_master[-1])
        
        if print_out:
            print("Rank warm start solution: ", avg_rank)
            print("Original average rank: ", avg_rank_DA)

        if print_out:
            if self.pricing.status == -1:
                if print_out:
                    print('Pricing problem INFEASIBLE')
            else:
                if print_out:
                    print('Objective pricing problem: ', self.obj_pricing[-1])

                    
        # Save the final solution
        # Create variables to store the solution in
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings
        zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
        self.Xassignment = Assignment(self.MyData, zero) # Contains the final assignment found by the model
        
        # Make sure assignment is empty in Xassignment
        self.Xassignment.assignment = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

        # Store decomposition
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings

        for l in self.N_MATCH:
            self.Xdecomp.append(np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools)))
            self.Xdecomp_coeff.append(self.w[l].varValue)
            for (i,j) in self.PAIRS:
                self.Xdecomp[-1][i,j] = self.M_list[l][i][j]
                self.Xassignment.assignment[i,j] += self.w[l].varValue * self.M_list[l][i][j]
                
        
        return self.Xassignment
    
    def generate_solution_report(self, print_out = False):
        # Store everything in a solution report
        S = SolutionReport()

        if self.time_limit_exceeded == False:  
            # Optimal solution of master problem!
            S.optimal = True
            S.time_limit_exceeded = False
            S.time = self.time_columnGen
            S.time_limit = self.time_limit
            if print_out:
                print("Optimal solution found!\nBest average rank: ", self.obj_master[-1])

        else:
            # Time limit exceeded
            S.optimal = False
            S.time_limit_exceeded = True
            S.time = self.time_limit
            S.time_limit = self.time_limit

            if print_out:
                print('\nTime limit of ', self.time_limit, "seconds exceeded!\n")
                print('Rank best found solution:', self.obj_master[-1])

        S.avg_ranks = {}
        S.avg_ranks['result'] = self.obj_master[-1]
        S.avg_ranks['first_iter']  = self.avg_rank_first_iter
        S.avg_ranks['warm_start'] = self.avg_rank
        S.avg_ranks['DA'] = self.avg_rank_DA

        if print_out:
            print("Rank first iteration: ", self.avg_rank_first_iter)
            print("Rank warm start solution: ", self.avg_rank)
            print("Original average rank: ", self.avg_rank_DA)

        S.obj_master = self.obj_master
        S.obj_pricing = self.obj_pricing

        # Save the final solution
        # Create variables to store the solution in
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings
        zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
        self.Xassignment = Assignment(self.MyData, zero) # Contains the final assignment found by the model
        
        # Make sure assignment is empty in Xassignment
        self.Xassignment.assignment = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

        # Store decomposition
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings

        for l in self.N_MATCH:
            self.Xdecomp.append(np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools)))
            self.Xdecomp_coeff.append(self.w[l].varValue)
            for (i,j) in self.PAIRS:
                self.Xdecomp[-1][i,j] = self.M_list[l][i][j]
                self.Xassignment.assignment[i,j] += self.w[l].varValue * self.M_list[l][i][j]
        
        S.Xdecomp = self.Xdecomp
        S.Xdecomp_coeff = self.Xdecomp_coeff
        S.A = copy.deepcopy(self.Xassignment)

        S.A_SIC = copy.deepcopy(self.p)
        S.A_DA_prob = copy.deepcopy(self.p_DA)

        S.iter = self.iterations

        return S
