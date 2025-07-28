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

from matplotlib.ticker import PercentFormatter

def SimulationsEvaluate(file_name: str, print_out = False):
    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)

    beta_in = 0.5
    # Average overall improvement in rank
    AvgRankImpr_percent(df, file_name, beta_in, print_out)
    #AvgRankImpr_absolute(df, file_name, beta_in, print_out)

    # Average individual improvement in rank (among improving agents)
    #AvgIndRankImpr(df, file_name, beta_in, print_out)

    # Display how the improvement differs with respect to n_stud
    #AvgImprovByNStud(df, file_name, beta_in, print_out)

    # Display by number of schools, one graph for each value of n_stud
    #AvgImprovByNSschools(df, file_name, beta_in, print_out)


def AvgRankImpr_percent(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    #print(df[df['beta']==1])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods):
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            legend.append("LP-heur EADA")

    # PERCENTAGE
    df_avg['DiffEE'] = (df_avg['avg_rank_DA'] - df_avg['avg_rank_EE'])/df_avg['avg_rank_DA'] # Difference in average rank Erdil & Ergin vs DA
    df_avg['DiffEADA'] = (df_avg['avg_rank_DA'] - df_avg['avg_rank_EADA'])/df_avg['avg_rank_DA'] # Difference in average rank EADA vs DA 

    counter = 1
    for s in labels:
        df_avg[f'DiffHeur{counter}'] = (df_avg['avg_rank_DA'] - df_avg[f'{counter}_avg_rank_heur'])/df_avg['avg_rank_DA']
        df_avg[f'DiffResult{counter}'] = (df_avg['avg_rank_DA'] - df_avg[f'{counter}_avg_rank_result'])/df_avg['avg_rank_DA']
        counter = counter + 1

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['DiffResult1'].max()

    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(5,4))
        plt.scatter(df_n['alpha'], df_n['DiffEE'], label = "Erdil & Ergin")
        plt.scatter(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            plt.scatter(df_n['alpha'], df_n[f'DiffHeur{counter}'], label = labels[counter - 1] + ' (heuristic)')
            # Check if better result column generation
            df_n['differs'] = (df_n[f'DiffHeur{counter}'] - df_n[f'DiffResult{counter}']).abs() >= 0.0001
            if df_n['differs'].any == True:
                 plt.scatter(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
            counter = counter  +1

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Improvement in expected rank')

        plt.ylim(-0.005, max_diff + 0.005)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Improvement in expected rank vs DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_percent_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")

def AvgRankImpr_absolute(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Transform ranks into pandas dataframe
    data = []
    for i in range(len(S_vector)):
        df2 = S_vector[i].avg_ranks.copy()
        df2['n_stud'] = S_vector[i].n_stud
        df2['n_schools'] = S_vector[i].n_schools
        df2['alpha'] = S_vector[i].alpha
        df2['beta'] = S_vector[i].beta

        data.append(df2)

    df = pd.DataFrame(data)

    #print(df[df['beta']==1])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean().reset_index()
        # Last command resets indices to keep using them as colummns

    # ABSOLUTE
    df_avg['DiffEE'] = (df_avg['DA'] - df_avg['warm_start']) # Difference in average rank Erdil & Ergin vs DA
    df_avg['DiffHeur'] = (df_avg['DA'] - df_avg['first_iter']) # Difference in average rank heuristic (first step column gen) vs DA
    df_avg['DiffCG'] = (df_avg['DA'] - df_avg['result']) # Difference in average rank column generation vs DA

    print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['DiffCG'].max()


    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(5,4))
        plt.scatter(df_n['alpha'], df_n['DiffEE'], label = "Erdil & Ergin")
        if S_vector[0].bool_ColumnGen == True:
            plt.scatter(df_n['alpha'], df_n['DiffCG'], label = "Column generation")
        plt.scatter(df_n['alpha'], df_n['DiffHeur'], label = "Heuristic")

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Improvement in expected rank')
        plt.ylim(-0.005, max_diff + 0.005)
        name_title = 'Improvement in expected rank vs DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_abs_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")


def AvgIndRankImpr(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    
    
    # Transform ranks into pandas dataframe
    data = []
    for i in range(len(S_vector)):
        comparison_US = S_vector[i].A.compare(S_vector[i].A_DA_prob, False)
        comparison_EE = S_vector[i].A_SIC.compare(S_vector[i].A_DA_prob, False)
        df2 = {}
        df2['n_stud'] = S_vector[i].n_stud
        df2['n_schools'] = S_vector[i].n_schools
        df2['alpha'] = S_vector[i].alpha
        df2['beta'] = S_vector[i].beta

        df2['n_improv_US'] = comparison_US['n_students_improving']
        df2['n_improv_EE'] = comparison_EE['n_students_improving']

        df2['average_rank_increase_US'] = comparison_US['average_rank_increase']
        df2['average_rank_increase_EE'] = comparison_EE['average_rank_increase']

        data.append(df2)

    df = pd.DataFrame(data)
    #print(df)

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean().reset_index()

    print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['average_rank_increase_US'].max() # Find max improvement


    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]

        plt.figure(figsize=(5,4))

        plt.scatter(df_n['alpha'], df_n['average_rank_increase_EE'], label = "Erdil & Ergin")
        if S_vector[0].bool_ColumnGen == True:
            plt.scatter(df_n['alpha'], df_n['average_rank_increase_US'], label = "Column generation")
        else: 
            plt.scatter(df_n['alpha'], df_n['average_rank_increase_US'], label = "Heuristic")
    

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Average individual improvement in expected rank')
        plt.ylim(-0.005, max_diff + 0.005)
        name_title = 'Individual improvement in expected rank vs DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/AvgIndRankImpr_abs_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")


def AvgImprovByNStud(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Transform ranks into pandas dataframe
    data = []
    for i in range(len(S_vector)):
        df2 = S_vector[i].avg_ranks.copy()
        df2['n_stud'] = S_vector[i].n_stud
        df2['n_schools'] = S_vector[i].n_schools
        df2['alpha'] = S_vector[i].alpha
        df2['beta'] = S_vector[i].beta

        data.append(df2)

    df = pd.DataFrame(data)

    #print(df[df['beta']==1])

    if len(df['n_stud'].unique()) > 1: # If you evaluated for more than a single number of students

        # Average out
        df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean().reset_index()
            # Last command resets indices to keep using them as colummns

        
        # ABSOLUTE
        df_avg['DiffEE'] = (df_avg['DA'] - df_avg['warm_start']) # Difference in average rank Erdil & Ergin vs DA
        df_avg['DiffHeur'] = (df_avg['DA'] - df_avg['first_iter']) # Difference in average rank heuristic (first step column gen) vs DA
        df_avg['DiffCG'] = (df_avg['DA'] - df_avg['result']) # Difference in average rank column generation vs DA

        df_avg = df_avg[df_avg['beta'] == beta_in]

        # Average over alpha and n_schools
        df_avg = df_avg.groupby(['n_stud']).mean().reset_index()
        print(df_avg)

        plt.figure(figsize=(5,4))

        plt.scatter(df_avg['n_stud'], df_avg['DiffEE'], label = "Erdil & Ergin")
        if S_vector[0].bool_ColumnGen == True:
            plt.scatter(df_avg['n_stud'], df_avg['DiffCG'], label = "Column generation")
        plt.scatter(df_avg['n_stud'], df_avg['DiffHeur'], label = "Heuristic")


        plt.xlabel("n")
        plt.legend()
        plt.ylabel('Improvement in expected rank')
        name_title = 'Improvement in expected rank by n vs DA\n (beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/RankImprByN_abs_beta_" + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")


def AvgImprovByNSschools(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Transform ranks into pandas dataframe
    data = []
    for i in range(len(S_vector)):
        df2 = S_vector[i].avg_ranks.copy()
        df2['n_stud'] = S_vector[i].n_stud
        df2['n_schools'] = S_vector[i].n_schools
        df2['alpha'] = S_vector[i].alpha
        df2['beta'] = S_vector[i].beta

        data.append(df2)

    df = pd.DataFrame(data)

    #print(df[df['beta']==1])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean().reset_index()
        # Last command resets indices to keep using them as colummns

    
    # ABSOLUTE
    df_avg['DiffEE'] = (df_avg['DA'] - df_avg['warm_start']) # Difference in average rank Erdil & Ergin vs DA
    df_avg['DiffHeur'] = (df_avg['DA'] - df_avg['first_iter']) # Difference in average rank heuristic (first step column gen) vs DA
    df_avg['DiffCG'] = (df_avg['DA'] - df_avg['result']) # Difference in average rank column generation vs DA

    df_avg = df_avg[df_avg['beta'] == beta_in]

    # Average over alpha 
    df_avg = df_avg.groupby(['n_schools']).mean().reset_index()
    print(df_avg)

    for i in df['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == i]

        plt.figure(figsize=(5,4))

        plt.scatter(df_n['n_schools'], df_n['DiffEE'], label = "Erdil & Ergin")
        if S_vector[0].bool_ColumnGen == True:
            plt.scatter(df_n['n_schools'], df_n['DiffCG'], label = "Column generation")
        plt.scatter(df_n['n_schools'], df_n['DiffHeur'], label = "Heuristic")


        plt.xlabel("number of schools")
        plt.legend()
        plt.ylabel('Improvement in expected rank')
        name_title = 'Improvement in expected rank by n vs DA\n (n = '+ str(i) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/RankImprByNSchools_abs_n_"+ str(i) + "_beta_" + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")


        plt.figure(figsize=(5,4))

        plt.scatter(df_n['n_schools'], df_n['warm_start'], label = "Erdil & Ergin")
        if S_vector[0].bool_ColumnGen == True:
            plt.scatter(df_n['n_schools'], df_n['result'], label = "Column generation")
        plt.scatter(df_n['n_schools'], df_n['first_iter'], label = "Heuristic")


        plt.xlabel("number of schools")
        plt.legend()
        plt.ylabel('Expected rank')
        name_title = 'Expected rank by n vs DA\n (n = '+ str(i) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/RankByNSchools_abs_n_"+ str(i) + "_beta_" + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")

