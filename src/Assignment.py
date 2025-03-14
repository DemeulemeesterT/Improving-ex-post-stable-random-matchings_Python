from .Data import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Assignment:
    """
    Class Assignment:
    This class models and manipulates an assignment of students to schools based on preferences and priorities.

    Functions:
    1. __init__(): Initializes the assignment with provided data and computes preference rankings.
    2. visualize(): Generates visualizations of the assignment and ranked preferences using heatmaps.
    3. export_assignment(): Saves the assignment data to a structured CSV file.
    """

    def __init__(self, MyData: Data, p: np.ndarray, M_set_in = None, label = None):
        """
        Initializes the assignment object with the given data, computes a ranked version of the assignment 
        based on student preferences, and exports the assignment to a CSV file.
        Arguments:
        - MyData (Data): The Data object containing student and school information.
        - p (np.ndarray): The assignment matrix indicating allocations of students to schools.
        - label (str, optional): A label for the assignment, used for file naming and visualization. Defaults to None.
        - M: the set of generated matchings (if any) to obtain the assignment
        """
        # self.file_name = MyData.file_name[:-4] 
            # Use this when importing .csv files, for example
        self.file_name = MyData.file_name
        self.MyData = copy.deepcopy(MyData)
        self.assignment = copy.deepcopy(p)
        self.M_set = copy.deepcopy(M_set_in)
        self.label = label
        if label == None:
            self.label = ""
        
        names = []
        for i in range(0,MyData.n_stud):
            names.append("Choice {}".format(i + 1))
        
        # Same as assignment, but ranked in decreasing order of preference
        self.assignment_ranked = np.zeros(shape=(MyData.n_stud, MyData.n_schools), dtype = np.float64)
        counter =  0
        for i in range(0, MyData.n_stud):
            for j in range(0, len(MyData.pref[i])):
                
                # Convert pref[i][k] (school ID as string) to column index
                #col_index = int(MyData.pref[i][j]) - 1
                col_index = MyData.ID_school.index(MyData.pref[i][j])
                self.assignment_ranked[i][j] = self.assignment[i][col_index]
                counter += 1
        #self.assignment_ranked = pd.DataFrame(ranked, columns = names)

    
        # Export assignment
        self.export_assignment()
    
    # Visualize the assignment in different ways
    def visualize(self):
        """
        Creates two heatmaps of the assignments: 
        - One that just depicts the assignment itself
        - One where the schools are ranked in decreasing order of preference for each student
        """ 

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
        sns.set(rc = {'figure.figsize':(self.MyData.n_stud,self.MyData.n_schools/1.5)})
        
        # Create a custom colormap (to show negative values red)
        colors = ["red", "white", "blue"]  # Red for negatives, white for 0, blue for positives
        custom_cmap = LinearSegmentedColormap.from_list("CustomMap", colors)
        
        # Create the heatmap
        p = sns.heatmap(self.assignment, cmap = custom_cmap, center=0, annot=True, yticklabels = self.MyData.ID_stud, xticklabels = self.MyData.ID_school)
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
        sns.set(rc = {'figure.figsize':(self.MyData.n_stud,self.MyData.n_schools/1.5)})
        p = sns.heatmap(self.assignment_ranked, cmap = custom_cmap2, center=0, annot=True, yticklabels = self.MyData.ID_stud, xticklabels = range(1,self.MyData.n_schools + 1))
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
    #def __str__(self):
        
    #    return s
        