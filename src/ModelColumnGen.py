from .Assignment import *
# Install necessary packages before running the script:
# pip install pulp
# pip install gurobipy
# pip install pyscipopt
# Note: Obtain an academic license for Gurobi from: https://www.gurobi.com/downloads/end-user-license-agreement-academic/

# Check with solvers available on computer
import pulp as pl
from pulp import *

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
        self.model = LpProblem("Improving_ex_post_stable_matchings", LpMinimize)

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
        self.M = {}
        for k in self.N_MATCH:
            for i in range(self.MyData.n_stud):
                for j in range(self.MyData.n_schools):
                    self.M[k, i, j] =self.M_list[k][i][j]  # Fill the parameter from the M_list

        #self.M = LpVariable.dicts("M", [(k, i, j) for k in self.N_MATCH for i, j in self.PAIRS], cat="Binary")

        # Store labels to make understanding output easier
        self.labels = {}
        for k, i, j in self.M:
            student_name = self.MyData.ID_stud[i]
            school_name = self.MyData.ID_school[j]
            self.labels[k, i, j] = f"M_{k}_{student_name}_{school_name}"

        #### OBJECTIVE FUNCTION ####
        # Add an empty objective function
            # Every time you update it, you should add it to the model again
                # using a code like:
                # self.model.setObjective(self.model.objective+obj_coeff*self.w[m])
        self.model += LpAffineExpression()
            
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

        # Add all these contraints to the model
        for c in self.constraints.values():
            self.model += c

        #### DECISION VARIABLES ####
        self.w = []
        print(self.nr_matchings)
        for m in self.N_MATCH:
            #print(m)
            # First, determine coefficients of this variable in the constraints
            coeff = {}
            coeff["Sum_to_one"] = 1
            coeff["Obj"] = 0 # Objective coefficients will be fixed later
            for i in self.STUD:
                for j in range(len(self.MyData.pref[i])):
                    school_name = self.MyData.pref_index[i][j]
                    name = "FOSD_" +  str(self.MyData.ID_stud[i]) + "_" + str(school_name)
                    
                    # Initialize the coefficient if it doesn't exist
                    if name not in coeff:
                        coeff[name] = 0
                    
                    for k in range(j+1):
                        pref_school = self.MyData.pref_index[i][k]
                        #print(m,i,j,pref_school, self.M[m,i,pref_school])
                        coeff[name] += self.M[m,i,pref_school]
        
            # Then, create a dictionary for `e` that maps constraints to their coefficients
            e_dict = {self.constraints[key]: coeff[key] for key in self.constraints if coeff[key] > 0}

            # Add this variable to self.w
            name_w = "w_" + str(m)
            self.w.append(LpVariable(name_w, lowBound=0, upBound=1, e=e_dict))
                          
            # Compute objective coefficient of this variable (average rank)
            obj_coeff = 0
            for (i,j) in self.PAIRS:
                obj_coeff += self.M[m,i,j]*(self.MyData.rank_pref[i,j]+ 1) / self.MyData.n_stud # + 1 because the indexing starts from zero

            # Add this variable to the model with the correct objective coefficient
            self.model.setObjective(self.model.objective+obj_coeff*self.w[m])



        # Create variables, and add them to the constraints
        
        # w[k] is the weight of matching k in the decomposition
        #self.w = LpVariable.dicts("w", self.N_MATCH, 0, 1,  )
        #self.w = LpVariable("w", self.nr_matchings, lowBound=0, upBound=1, e={self.constraints["Sum_to_one"]:1} )

        #self.w = LpVariable("w", self.nr_matchings, e={self.constraints["Sum_to_one"]:1} )
        #self.model += 2*self.w

        
        #self.vars += self.w()
        #for m in self.N_MATCH:
        #    for c in range(1,len(self.constraints)):
        #        self.model.addVariableToConstraints(self.w[m], {self.constraints[c], 1})

        self.model.writeLP("TestColumnFormulation.lp")

                                    