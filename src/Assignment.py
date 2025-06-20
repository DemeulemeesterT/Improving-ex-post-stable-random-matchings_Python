from .Data import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics

class Assignment:
    """
    Class Assignment:
    This class models and manipulates an assignment of students to schools based on preferences and priorities.

    Functions:
    1. __init__(): Initializes the assignment with provided data and computes preference rankings.
    2. visualize(): Generates visualizations of the assignment and ranked preferences using heatmaps.
    3. statistics(): Calculate statistics about the assignment, such as average rank
    4. export_assignment(): Saves the assignment data to a structured CSV file.
    """

    def __init__(self, MyData: Data, p: np.ndarray, M_set_in = None, w_set_in = None, label = None):
        """
        Initializes the assignment object with the given data, computes a ranked version of the assignment 
        based on student preferences, and exports the assignment to a CSV file.
        Arguments:
        - MyData (Data): The Data object containing student and school information.
        - p (np.ndarray): The assignment matrix indicating allocations of students to schools.
        - label (str, optional): A label for the assignment, used for file naming and visualization. Defaults to None.
        - M_set_in: the set of generated matchings (if any) to obtain the assignment
        - w_set_in: the set of weights used for the matchings to obtain the assignment (if any)
            (w_set should be a dictionary)
        """
        # self.file_name = MyData.file_name[:-4] 
            # Use this when importing .csv files, for example
        self.file_name = MyData.file_name
        self.MyData = copy.deepcopy(MyData)
        self.assignment = copy.deepcopy(p)
        self.M_set = copy.deepcopy(M_set_in)
        self.w_set = copy.deepcopy(w_set_in)
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
    
    def statistics(self, print_out = False):
        # Calculate statistics about the assignment (such as average rank)

        # Tuple with all student-school pairs that are preferred to outside option
        # This tuple contains the INDICES of the students and the pairs, and not their original names
        PAIRS = []
        for i in range(0, self.MyData.n_stud):
            for j in range(0,len(self.MyData.pref[i])):
                # Convert pref[i][k] (school ID as string) to column index
                col_index = self.MyData.ID_school.index(self.MyData.pref[i][j])
                PAIRS.append((i,col_index))   

        # Compute average rank of current assignment
        avg_rank = 0
        for (i,j) in PAIRS:
            avg_rank += self.assignment[i,j] * (self.MyData.rank_pref[i,j] + 1) # + 1 because the indexing starts from zero
        # Average
        avg_rank = avg_rank/self.MyData.n_stud
        if print_out == True:   
            print(f"\nAverage rank: {avg_rank}.\n")

        return avg_rank

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
        


    def compare(self, benchmark: np.ndarray, print_out = False):
        """
        Compares the obtained assignment with respect to a benchmark assignment on several criteria
        """

        # Number of agents with improvements
        students_improving = [False] * self.MyData.n_stud
        for i in range(self.MyData.n_stud):
            sd_dominating = True # Will become False if new is not sd-dominating old
            sum_new = 0
            sum_old = 0
            better = False # True if new is better at least at some point

            for j in range(len(self.MyData.pref[i])):
                school = self.MyData.pref_index[i][j]
                sum_new = sum_new + self.assignment[i][school]
                sum_old = sum_old + benchmark[i][school]
                if sum_new > sum_old + 0.00001: 
                    better = True
                elif sum_new < sum_old - 0.00001:
                    sd_dominating = False
                    #print('Careful, student ', i, ' improves somewhere, but is not sd-dominating!', j, sum_new, sum_old )

            
            if (better == True) and (sd_dominating == True):
                students_improving[i] = True
                #print('Improving agent', i)

        # Number of improving agents
        n_students_improving = sum(students_improving)

        # Expected increase in rank for the students
        rank_increase_by_stud = [0] * self.MyData.n_stud
        for i in range(self.MyData.n_stud):
            if students_improving[i]:
                old_rank = 0
                new_rank = 0
                for j in range(len(self.MyData.pref[i])):
                    school = self.MyData.pref_index[i][j]
                    old_rank = old_rank + (j+1) * benchmark[i][school]
                    new_rank = new_rank + (j+1) * self.assignment[i][school]
                rank_increase_by_stud[i] = old_rank - new_rank

        # Average increase in rank, among improving students
        average_rank_increase = 0
        for i in range(self.MyData.n_stud):
            if students_improving[i]:
                average_rank_increase = rank_increase_by_stud[i] + average_rank_increase
        if n_students_improving > 0:
            average_rank_increase = average_rank_increase / n_students_improving
        else:
            average_rank_increase = 0


        # Median increase in rank, among improving students
        rank_improvements = []
        for i in range(self.MyData.n_stud):
            if students_improving[i]:
                rank_improvements.append(rank_increase_by_stud[i])
        if n_students_improving > 0:
            median_rank_improvement = statistics.median(rank_improvements)
        else:
            median_rank_improvement = 0

        if print_out:
            print("Number of improving studentss",n_students_improving)
            print("Average rank improvement", average_rank_increase)
            print("Median rank improvement", median_rank_improvement)


        return {"students_improving": students_improving, "n_students_improving": n_students_improving,
                'rank_increase_by_stud': rank_increase_by_stud, "average_rank_increase": average_rank_increase,
                'median_rank_improvement': median_rank_improvement}
    


