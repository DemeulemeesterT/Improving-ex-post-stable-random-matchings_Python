import numpy as np
#import pandas as pd
import copy # To make deep copies
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


## Define classes
#Define the following classes:
#* 'Data': contains
#    * Number of students
#    * Number of schools
#    * Preferences students
#    * Preferences schools
#    * Capacities schools
#    * Names of students
#    * Names of schools
#    * File name
#* 'Assignment': the selection probabilities of the students to the schools

class Data:
    # Define the initialization of an object from this class
    def __init__(self, n_stud: int, n_schools: int, pref: list, prior: list, cap:list, ID_stud:list, ID_school:list, file_name:str):
        self.n_stud = n_stud
        self.n_schools = n_schools
        self.pref = copy.deepcopy(pref)
        self.prior = copy.deepcopy(prior)
        self.cap = copy.deepcopy(cap)
        self.ID_stud = copy.deepcopy(ID_stud)
        self.ID_school = copy.deepcopy(ID_school)
        self.file_name = file_name

        

    # Choose what is being shown for the command 'print(MyData)', where 'MyData' is an instance of the class 'Data'
    def __str__(self):
        s ="The data instance has the following properties: \n"
        s += f"\n\t{self.n_stud} students.\n\t{self.n_schools} schools. \n\n \tPREFERENCES:\n"
        for i in range(0,self.n_stud):
            s+= f"\t{self.ID_stud[i]}\t"
            for j in range(0, len(self.pref[i])):
                if len(self.pref[i][j]) >= 2:
                    s+=f"{{"
                    for k in range(0, len(self.pref[i][j])):
                        s+=f"{self.pref[i][j][k]}"
                        if k < len(self.pref[i][j]) - 1:
                            s+= f" "
                    s+=f"}} "
                else:
                    s+=f"{self.pref[i][j]} "
            s +="\n"

        s += f"\n\n \tCAPACITIES & PRIORITIES:\n"
        for i in range(0,self.n_schools):
            s+= f"\t{self.ID_school[i]}\t"
            s+= f"{self.cap[i]}\t"
            for j in range(0, len(self.prior[i])):
                if len(self.prior[i][j]) >= 2:
                    s+=f"{{"
                    for k in range(0, len(self.prior[i][j])):
                        s+=f"{self.prior[i][j][k]}"
                        if k < len(self.prior[i][j]) - 1:
                            s+= f" "
                    s+=f"}} "
                else:
                    s+=f"{self.prior[i][j]} "
            s +="\n"
        return s


class Assignment:
    # This class will contain an assignment
    def __init__(self, MyData: Data, p: np.ndarray, label = None):
        # self.file_name = MyData.file_name[:-4] 
            # Use this when importing .csv files, for example
        self.file_name = MyData.file_name
        self.MyData = copy.deepcopy(MyData)
        self.assignment = copy.deepcopy(p)
        self.label = label
        
        names = []
        for i in range(0,MyData.n_stud):
            names.append("Choice {}".format(i + 1))
        
        # Same as assignment, but ranked in decreasing order of preference
        self.assignment_ranked = np.zeros(shape=(MyData.n_stud, MyData.n_schools), dtype = np.float64)
        counter =  0
        for i in range(0, MyData.n_stud):
            for j in range(0, len(MyData.pref[i])):
                
                # Convert pref[i][k] (school ID as string) to column index
                col_index = int(MyData.pref[i][j]) - 1
                self.assignment_ranked[i][j] = self.assignment[i][col_index]
                counter += 1
        #self.assignment_ranked = pd.DataFrame(ranked, columns = names)

    
        # Export assignment
        self.export_assignment()
    
    # Visualize the assignment in different ways
    def visualize(self):
        # To export the figures, check if the correct folder exists:
        if os.path.exists("Results") == False:
            # If not, create folder
            os.makedirs("Results")
        
        s = os.path.join("Results", "Visualisations")
        if os.path.exists(s) == False:
            # If not, create folder
            os.makedirs(s)
        
        s = os.path.join("Results", "Visualisations",self.file_name)
        if os.path.exists(s) == False:
            os.makedirs(s)
            
        
        path = "Results/Visualisations/"
        # The assignment itself
        sns.set(rc = {'figure.figsize':(MyData.n_stud,MyData.n_schools/1.5)})
        
        # Create a custom colormap (to show negative values red)
        colors = ["red", "white", "blue"]  # Red for negatives, white for 0, blue for positives
        custom_cmap = LinearSegmentedColormap.from_list("CustomMap", colors)
        
        # Create the heatmap
        p = sns.heatmap(self.assignment, cmap = custom_cmap, center=0, annot=True, yticklabels = MyData.ID_stud, xticklabels = MyData.ID_school)
        p.set_xlabel("Students", fontsize = 15)
        p.set_ylabel("Schools", fontsize = 15)
        name = path + self.file_name + "/" + self.label + ".pdf"
        p.set_title(self.label, fontsize = 20)
        plt.savefig(name, format="pdf", bbox_inches="tight")
        
        # Assignment, ranked by preference
        plt.figure()

        # Create a custom colormap (to show negative values red)
        colors = ["red", "white", "green"]  # Red for negatives, white for 0, blue for positives
        custom_cmap2 = LinearSegmentedColormap.from_list("CustomMap", colors)
        
        # Create the heatmap
        sns.set(rc = {'figure.figsize':(MyData.n_stud,MyData.n_schools/1.5)})
        p = sns.heatmap(self.assignment_ranked, cmap = custom_cmap2, center=0, annot=True, yticklabels = MyData.ID_stud, xticklabels = range(1,MyData.n_schools + 1))
        p.set_xlabel("Preference", fontsize = 15)
        p.set_ylabel("Students", fontsize = 15)
        name = path + self.file_name + "/" + self.label + "_Ranked.pdf"
        title = self.file_name + ": ranked by decreasing preference"
        p.set_title(title, fontsize = 20)
        plt.savefig(name, format="pdf", bbox_inches="tight")
        
        plt.figure()
    
    # Save the assignment to the correct subdirectory
    def export_assignment(self):
        if os.path.exists("Results") == False:
            # If not, create folder
            os.makedirs("Results")

        s = os.path.join("Results", "Assignments")
        if os.path.exists(s) == False:
            # If not, create folder
            os.makedirs(s)

        s = os.path.join("Results", "Assignments",self.file_name)
        if os.path.exists(s) == False:
            os.makedirs(s)
        
        name = "Results/Assignments/" + self.file_name + "/" + self.label + "_" + self.file_name + ".csv"
        np.savetxt(name, self.assignment, delimiter=",")
        
    # Choose what is being shown for the command 'print(Sol)', where 'Sol' is an instance of the class 'Assignment'
    def __str__(self):
        
        return s
        