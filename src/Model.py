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

class Model: 
    """
    Contains two methods:
        __init__: initializes the model, and the solver environment

        Solve: solves the model.
            The parameters of this method can control which objective function is optimized, and which solver is used
    """
    
    # Used this example as a template for Pulp: https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html
    
    def __init__(self, MyData: Data, p: Assignment, print_out: bool, nr_matchings = -1):
        """
        Initialize an instance of Model.

        Args:
            MyData (type: Data): instance of class Data.
            p (type: Assignment): instance of class Assignment.
            print_out (type: bool): boolean that controls which output is printed.
            nr_matchings (optional): number of matchings used in the decomposition, optional parameter that defaults to n_students * n_schools + 1

        """
        # 'nr_matchings' refers to number of matchings used to find decomposition
        self.MyData = copy.deepcopy(MyData)
        self.p = copy.deepcopy(p)
        self.nr_matchings = nr_matchings
        if nr_matchings == -1:
            self.nr_matchings = self.MyData.n_stud * self.MyData.n_schools + 1

        # Create the pulp model
        self.model = LpProblem("Improving_ex_post_stable_matchings", LpMinimize)

        # Create variables to store the solution in
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings
        zero = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))
        self.Xassignment = Assignment(MyData, zero) # Contains the final assignment found by the model

        #### DECISION VARIABLES ####
        self.STUD = range(0,self.MyData.n_stud)
        self.SCHOOLS = range(0, self.MyData.n_schools)
        self.N_MATCH = range(0, self.nr_matchings)

        # Tuple with all student-school pairs that are preferred to outside option
        # This tuple contains the INDICES of the students and the pairs, and not their original names
        self.PAIRS = []
        for i in range(0, MyData.n_stud):
            for j in range(0,len(MyData.pref[i])):
                # Convert pref[i][k] (school ID as string) to column index
                col_index = MyData.ID_school.index(MyData.pref[i][j])
                self.PAIRS.append((i,col_index))   
        
        # M[k][i][j] = 1 if student i is assigned to school j in matching k, and 0 otherwise
        self.M = LpVariable.dicts("M", [(k, i, j) for k in self.N_MATCH for i, j in self.PAIRS], cat="Binary")

        # Auxiliary variables to avoid non-linearity
        self.z = LpVariable.dicts("z", [(k, i, j) for k in self.N_MATCH for (i, j) in self.PAIRS], 0, 1)

        # Rename M and z
        for k, i, j in self.M:
            student_name = self.MyData.ID_stud[i]
            school_name = self.MyData.ID_school[j]
            self.M[k, i, j].name = f"M_{k}_{student_name}_{school_name}"
            self.z[k, i, j].name = f"z_{k}_{student_name}_{school_name}"

        # Q[i][j] is the new probability with which student i is assigned to school j, lies between 0 and 1
        self.Q = LpVariable.dicts("q", self.PAIRS, 0, 1) 
    
        # w[k] is the weight of matching k in the decomposition
        self.w = LpVariable.dicts("w", self.N_MATCH, 0, 1)

        #### OBJECTIVE FUNCTION ####
            # Done separately in other functions (see function Solve)
        
            
        #### CONSTRAINTS ####
        # Other constraints defined for specific models in functions below (see function Solve)

        # Stability
        for k in self.N_MATCH:
            for i in self.STUD:
                for j in range(len(self.MyData.pref_index[i])):
                    current_school = self.MyData.pref_index[i][j]
                    lin = LpAffineExpression()

                    lin += self.MyData.cap[current_school] * self.M[k, i, current_school]

                    # Add all schools that are at least as preferred as the j-ranked school by student i
                    for l in range(j):
                        lin += self.MyData.cap[current_school] * self.M[k,i,self.MyData.pref_index[i][l]]


                    # Add terms based on priorities
                    prior_current = self.MyData.rank_prior[current_school][i]
                    for s in self.STUD:
                        if s != i:
                            # If current_school ranks student s higher than student i
                            if self.MyData.rank_prior[current_school][s] <= self.MyData.rank_prior[current_school][i]:
                                if (s, current_school) in self.PAIRS:
                                    lin += self.M[k,s,current_school]

                    # Add to model:
                    name = "STAB_" + str(k) + "_" + str(self.MyData.ID_stud[i]) + "_" + str(self.MyData.ID_school[current_school]) 
                    self.model += (lin >= self.MyData.cap[current_school], name) 

        
        # Each student at most assigned to one school
        for l in self.N_MATCH:
            for i in self.STUD:
                self.model += lpSum([self.M[l,i,j] for j in self.SCHOOLS if (i,j) in self.PAIRS]) <= 1, f"LESS_ONE_{l,i}"

        # Capacities schools respected
        for l in self.N_MATCH:
            for j in self.SCHOOLS:
                self.model += lpSum([self.M[l,i,j] for i in self.STUD if (i,j) in self.PAIRS]) <= self.MyData.cap[j], f"LESS_CAP_{l,j}"
                                    

    def Solve(self, obj: str, solver: str, print_out: bool):
        """
        Solves the formulation.
        Returns an instance from the Assignment class.

        Args:
            obj (str): controls the objective function
                "IMPR_RANK": minimizes expected rank while maintaining ex-post stability
                "STABLE": maximizes fraction of stable matchings in decomposition
                "EX_ANTE": finds ex-ante stable improvement (heuristic)
            solver (str): controls which solver is used. See options through following commands:
                solver_list = pl.listSolvers(onlyAvailable=True)
                print(solver_list)
            print_out (bool): boolean that controls which output is printed.

        """

        # Check that strings-arguments are valid

        # Valid values for 'solver'
        solver_list = pl.listSolvers(onlyAvailable=True)
        if solver not in solver_list:
           raise ValueError(f"Invalid value: '{solver}'. Allowed values are: {solver_list}")

        # Valid values for 'obj'
        obj_list = ["IMPR_RANK", "STABLE"]
        if obj not in obj_list:
           raise ValueError(f"Invalid value: '{obj}'. Allowed values are: {obj_list}")

        #### FORMULATION ####
        
        # Set the objective function
        if obj == "IMPR_RANK":
            self.Improve_rank(print_out)
        
        elif obj == "STABLE":
            self.Max_Stable_Fraction(print_out)

        self.model.writeLP("TestFormulation.lp")

        
        #### SOLVE ####
            
        # String can't be used as the argument in solve method, so convert it like this:
        solver_function = globals()[solver]  # Retrieves the GUROBI function or class
        
        # Solve the formulation
        self.model.solve(solver_function())
        #self.model.solve(GUROBI_CMD(keepFiles=True, msg=True, options=[("IISFind", 1)]))
        
        #### STORE SOLUTION ####
        # Make sure assignment is empty in Xassignment
        self.Xassignment.assignment = np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools))

        for (i,j) in self.PAIRS:
            self.Xassignment.assignment[i,j] = self.Q[i,j].varValue

        # Store decomposition
        self.Xdecomp = [] # Matchings in the found decomposition
        self.Xdecomp_coeff = [] # Weights of these matchings

        for l in self.N_MATCH:
            self.Xdecomp.append(np.zeros(shape=(self.MyData.n_stud, self.MyData.n_schools)))
            self.Xdecomp_coeff.append(self.w[l].varValue)
            for (i,j) in self.PAIRS:
                self.Xdecomp[-1][i,j] = self.M[l,i,j].varValue
                
        return self.Xassignment


    def Improve_rank(self, print_out: bool):
        """
        Creates and solves formulation to minimize the expected rank while ensuring the found random matching is ex-post stable.
        """
        
        if print_out == True:
            # Compute average rank of current assignment

            sum = 0
            for (i,j) in self.PAIRS:
                sum += self.p.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
            # Average
            sum = sum/self.MyData.n_stud
            print(f"\nAverage rank before optimization: {sum}.\n\n")
        
        # Objective function
        lin = LpAffineExpression()
        for (i,j) in self.PAIRS:
            lin += (self.Q[i,j] * (self.MyData.rank_pref[i,j] + 1)) / self.MyData.n_stud # + 1 because the indexing starts from zero
        self.model += lin

        # Define q based on matchings in decomposition
            # Where z is an auxiliary variable to avoid non-linearities
        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.w[l] <= 0,f"z_w{l,i,j}" 

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.M[l, i, j] <= 0,f"z_M_{l, i, j}"

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] + (1 - self.M[l, i, j]) - self.w[l]  >= 0,f"z_w_M_{l, i, j}"
                # Maybe these constraints are redundant because of the objective function

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) == self.Q[i,j], f"z_Q_{i, j}"

        # Ensure weights sum up to one
        self.model += lpSum([self.w[l] for l in self.N_MATCH]) == 1, f"SUM_TO_ONE"

        # First-order stochastic dominance
        for i in self.STUD:
            for j in range(len(self.MyData.pref[i])):
                lin = LpAffineExpression()
                for k in range(j+1):
                    pref_school = self.MyData.pref_index[i][k]
                    lin += self.Q[i,pref_school]
                    lin -= self.p.assignment[i,pref_school]
                name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(j)
                self.model += (lin >= 0, name)


    def Max_Stable_Fraction(self, print_out: bool):
        # Objective function
        obj = LpAffineExpression()
        for l in self.N_MATCH:
            obj += self.w[l] 
        self.model += obj
        self.model.sense = LpMaximize

        # Constraints to ensure that decomposition is at least equal to p (element-wise)
            # Where z is an auxiliary variable to avoid non-linearities
        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.w[l] <= 0,f"z_w{l,i,j}" 

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] - self.M[l, i, j] <= 0,f"z_M_{l, i, j}"

        for l in self.N_MATCH:
            for (i,j) in self.PAIRS:
                self.model += self.z[l, i, j] + (1 - self.M[l, i, j]) - self.w[l]  >= 0,f"z_w_M_{l, i, j}"
                # Maybe these constraints are redundant because of the objective function

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) == self.Q[i,j], f"z_Q_{i, j}"

        for (i,j) in self.PAIRS:
            self.model += lpSum([self.z[l, i, j] for l in self.N_MATCH]) <= self.p.assignment[i,j], f"z_p_{i, j}"


    def print_solution(self):
        s = "The obtained random matching is:\n"
        s+=f"\t\t"
        for j in self.SCHOOLS:
            s+=f"{self.MyData.ID_school[j]}\t"
        s+="\n"
        for i in self.STUD:
            s+= f"\t{self.MyData.ID_stud[i]}\t"
            for j in self.SCHOOLS:
                s+=f"{self.Xassignment.assignment[i,j]}\t"
            s+=f"\n"
        s+=f"\n"

        s+= "The matchings with positive weights are:\n"

        for l in self.N_MATCH:
            if self.Xdecomp_coeff[l] > 0:
                s+=f"\t w[{l}] = {self.Xdecomp_coeff[l]}\n"
                for i in self.STUD:
                    s+=f"\t\t"
                    for j in self.SCHOOLS:
                        if self.Xdecomp[l][i,j] == 1:
                            s+=f"1\t"
                        else:
                            s+= f"0\t"
                    s+=f"\n"
                s+=f"\n"
        print(s)

        
                
        