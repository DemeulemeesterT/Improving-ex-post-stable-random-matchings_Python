from .Assignment import *
# Install necessary packages before running the script:
# pip install pulp
# pip install gurobipy
# pip install pyscipopt
# Note: Obtain an academic license for Gurobi from: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

# Check with solvers available on computer
import pulp as pl
from pulp import *

import gurobipy

# To check which solvers available on computer:
# print(pl.listSolvers(onlyAvailable=True))


class ModelColumnGen: 
    """
    Contains two methods:
        __init__: initializes the model, and the solver environment

        Solve: solves the model.
            The parameters of this method can control which objective function is optimized, and which solver is used
    """
    
    # Used this example as a template for Pulp: https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
    
    def __init__(self, MyData: Data, p: Assignment, print_out: bool):
        """
        Initialize an instance of Model.

        Args:
            MyData (type: Data): instance of class Data.
            p (type: Assignment): instance of class Assignment.
            print_out (type: bool): boolean that controls which output is printed.
            nr_matchings (optional): number of matchings used in the decomposition, optional parameter that defaults to n_students * n_schools + 1

        """
        self.MyData = copy.deepcopy(MyData)
        self.p = copy.deepcopy(p)

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
                    original_p += self.p.assignment[i,pref_school]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                self.constraints[name]=LpConstraintVar(name, LpConstraintGE, original_p)

        # Non-negativity (explicitly included to get dual variables)
        for m in self.N_MATCH:
            name = 'GE0_' + str(m)
            self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
        
        # Add all these contraints to the model
        for c in self.constraints.values():
            self.master += c

        

        #### DECISION VARIABLES ####
        self.w = []
        #print(self.nr_matchings)
        for m in self.N_MATCH:
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

        self.master.writeLP("TestColumnFormulation.lp")
        
        
      
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
        for m in self.N_MATCH: 
            name = 'GE0_' + str(m)
            coeff[name] = 0
        name = 'GE0_' + str(index)
        coeff[name] = 1
        
        # Then, create a dictionary for `e` that maps constraints to their coefficients
        e_dict = {self.constraints[key]: coeff[key] for key in self.constraints if coeff[key] > 0}

        # Add this variable to self.w
        name_w = "w_" + str(index)
        self.w.append(LpVariable(name_w, e=e_dict))
                        
        # Compute objective coefficient of this variable (average rank)
        obj_coeff = 0
        for (i,j) in self.PAIRS:
            obj_coeff += M_in[i,j]*(self.MyData.rank_pref[i,j]+ 1) / self.MyData.n_stud # + 1 because the indexing starts from zero

        # Add this variable to the model with the correct objective coefficient
        self.master.setObjective(self.master.objective+obj_coeff*self.w[index])

        
    def Solve(self, stab_constr: str, solver: str, print_log: str, print_out: bool):
        """
        Solves the formulation using column generation.
        Returns an instance from the Assignment class.

        Args:
            stab_constr (str): controls which type of stability constraints are used.
            solver (str): controls which solver is used. See options through following commands:
                solver_list = pl.listSolvers(onlyAvailable=True)
                print(solver_list)
            print_out (bool): boolean that controls which output is printed.
        """
        # Compute average rank of current assignment
        avg_rank = 0
        for (i,j) in self.PAIRS:
            avg_rank += self.p.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
        # Average
        avg_rank = avg_rank/self.MyData.n_stud
        #if print_out == True:   
        print(f"\nAverage rank before optimization: {avg_rank}.\n\n")
        
        
        # Check that strings-arguments are valid

        # Valid values for 'solver'
        solver_list = pl.listSolvers(onlyAvailable=True)
        if solver not in solver_list:
           raise ValueError(f"Invalid value: '{solver}'. Allowed values are: {solver_list}")

        # Valid values for 'stab_constr'
        stab_list = ["TRAD"]
        if stab_constr not in stab_list:
           raise ValueError(f"Invalid value: '{stab_constr}'. Allowed values are: {stab_list}")        


        #### PRICING PROBLEM ####
        # Defined in separate function
        self.build_pricing(stab_constr, print_out)

       
        
        #### RUN COLUMN GENERATION PROCEDURE ####
        optimal = False
        
        self.pricing.writeLP("PricingProblem.lp")
        iterations = 1
        print("Number of matchings:", self.nr_matchings)

        while (optimal == False):
            print('ITERATION:', iterations)            
            if print_out:
                print("\n ****** MASTER ****** \n")
                for m in self.N_MATCH:
                    print(self.M_list[m])


            #### SOLVE MASTER ####
            # Create two empty arrays to store objective values of master and pricing problem
            self.obj_master = []
            self.obj_pricing = []

            # String can't be used as the argument in solve method, so convert it like this:
            solver_function = globals()[solver]  # Retrieves the GUROBI function or class
            
            # Solve the formulation
            if print_log == False:
                self.master.solve(solver_function(msg = False, logPath = "Logfile_master.log"))
            else:
                self.master.solve(solver_function(msg = True))
            #self.model.solve(GUROBI_CMD(keepFiles=True, msg=True, options=[("IISFind", 1)]))
            
            # Get objective value master problem
            self.obj_master.append(self.master.objective.value())
            print("Objective master: ", self.obj_master[-1])

            if print_out:
                for m in self.N_MATCH:
                    print("w_", m, self.w[m].value())
            
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

            for m in self.N_MATCH:
                name_GE = 'GE0_' + str(m)
                duals[name_GE] = self.master.constraints[name_GE].pi
            if print_out:
                print(duals)
                
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

            if print_out:
                print(pricing_obj)
            #print("Duals Sum_to_one", duals["Sum_to_one"])
            #print("Duals", duals)


            self.pricing.setObjective(pricing_obj)


            
            self.pricing.writeLP("PricingProblem.lp")
            
            # Solve modified pricing problem
            if print_out:
                print("\n ****** PRICING ****** \n")

            if print_log == True:
                self.pricing.solve(solver_function())
            else:
                self.pricing.solve(solver_function(msg=False, logPath = 'Logfile_pricing.log'))

            # Add the constant terms!
            constant = 0
            constant += duals['Sum_to_one']
                # This constant term will not be printed in the objective function in the .lp file, I think

            for m in self.N_MATCH:
                name_GE = 'GE0_' + str(m)
                constant += duals[name_GE]

            # If not infeasible:
            if self.pricing.status != -1:
                #print("Status code: ", self.pricing.status)
                obj_pricing_var = self.pricing.objective.value() + constant
                self.obj_pricing.append(obj_pricing_var)
                print("\t\tObjective pricing: ", obj_pricing_var)

            #if print_out:
            if False:
                for (i,j) in self.PAIRS:
                    print("M[",i,j,'] =', self.M_pricing[i,j].value())
            
            #### EVALUATE SOLUTION ####
            if self.pricing.status != -1:            
                if obj_pricing_var < 0:
                    # The solution of the master problem is not optimal over all weakly stable matchings
                    
                    # Add non-negativity constraint to the master for this new matching
                    name = 'GE0_' + str(len(self.w))
                    self.constraints[name] = LpConstraintVar(name, LpConstraintGE, 0)
                    self.master += self.constraints[name]
                    
                    # Add the matching found by the pricing problem to the master problem       
                    found_M = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
                    for (i,j) in self.PAIRS:
                        found_M[i][j] = self.M_pricing[i,j].value()
                    
                    self.add_matching(found_M, len(self.w), print_out)
                    
                    self.M_list.append(found_M)
                    self.nr_matchings += 1
                    self.N_MATCH = range(self.nr_matchings)
                    
                    print("Matching added.")
                    if print_out:
                        print(found_M)

                    if iterations == 10:
                        optimal = True
                        print("Process terminated after ", iterations, " iterations.")
                    iterations = iterations + 1                
            
                else:
                    optimal = True  
                    return self.pricing_opt_solution(avg_rank, print_out)     
            else:
                optimal = True
                return self.pricing_opt_solution(avg_rank, print_out)     


            

    def build_pricing(self, stab_constr: str, print_out: bool):
        # Create Pulp model for pricing problem
        self.pricing = LpProblem("Pricing problem", LpMinimize)

        # Decision variables
        self.M_pricing = LpVariable.dicts("M", [(i, j) for i, j in self.PAIRS], cat="Binary")
        # Rename M
        for i, j in self.M_pricing:
            student_name = self.MyData.ID_stud[i]
            school_name = self.MyData.ID_school[j]
            self.M_pricing[i, j].name = f"M_{student_name}_{school_name}"

        ### CONSTRAINTS ###
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

        
        # Each student at most assigned to one school
        for i in self.STUD:
            self.pricing += lpSum([self.M_pricing[i,j] for j in self.SCHOOLS if (i,j) in self.PAIRS]) <= 1, f"LESS_ONE_{l,i}"

        # Capacities schools respected
        for j in self.SCHOOLS:
            self.pricing += lpSum([self.M_pricing[i,j] for i in self.STUD if (i,j) in self.PAIRS]) <= self.MyData.cap[j], f"LESS_CAP_{l,j}"
         
        # Exclude matchings that are already found:
        # Simple "no-good" cuts, where you sum matched student-school pairs for matching l, and force the sum to be strictly smaller
        #for l in self.N_MATCH:
            #lin = LpAffineExpression()
            #counter = 0
            #for (i,j) in self.PAIRS:
            #    if self.M_list[l][i][j] == 1:
            #        lin += self.M_pricing[i,j]
            #        counter = counter + 1
            #self.pricing += (lin <= counter - 1, f"EXCL_M_{l}")
            
            #self.pricing += lpSum([self.M_pricing[i,j] * self.M_list[l][i][j] for (i,j) in self.PAIRS]) <= lpSum([self.M_list[l][i][j] for (i,j) in self.PAIRS]) - 1, f"EXCL_M_{l}"
        

    def pricing_opt_solution(self, avg_rank: int, print_out: str):
        # Optimal solution of master problem!
        print("Optimal solution found! Best average rank: ", self.obj_master[-1])
        print("Original average rank: ", avg_rank)

        if print_out:
            if self.pricing.status == -1:
                print('Pricing problem INFEASIBLE')
            else:
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
                self.Xdecomp[-1][i,j] = self.M[l,i,j]
                self.Xassignment.assignment[i,j] += self.w[l].varValue * self.M[l,i,j]
                
        
        return self.Xassignment