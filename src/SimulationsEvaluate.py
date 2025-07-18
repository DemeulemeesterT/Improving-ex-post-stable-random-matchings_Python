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
import pandas as pd
from matplotlib.ticker import PercentFormatter

def SimulationsEvaluate(S_vector: list[SolutionReport], name: str, print_out = False):
    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + name
    os.makedirs(name_folder, exist_ok=True)

    beta_in = 0.5
    # Average overall improvement in rank
    #AvgRankImpr_percent(S_vector, name, beta_in, print_out)
    #AvgRankImpr_absolute(S_vector, name, beta_in, print_out)

    # Average individual improvement in rank (among improving agents)
    AvgIndRankImpr(S_vector, name, beta_in, print_out)

def AvgRankImpr_percent(S_vector: list[SolutionReport], name: str, beta_in:float, print_out = False):
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

    # PERCENTAGE
    df_avg['DiffEE'] = (df_avg['DA'] - df_avg['warm_start'])/df_avg['DA'] # Difference in average rank Erdil & Ergin vs DA
    df_avg['DiffHeur'] = (df_avg['DA'] - df_avg['first_iter'])/df_avg['DA'] # Difference in average rank heuristic (first step column gen) vs DA
    df_avg['DiffCG'] = (df_avg['DA'] - df_avg['result']) / df_avg['DA']# Difference in average rank column generation vs DA

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]
    plt.figure(figsize=(5,4))
    plt.scatter(df_avg['alpha'], df_avg['DiffEE'], label = "Erdil & Ergin vs DA")
    plt.scatter(df_avg['alpha'], df_avg['DiffCG'], label = "Column Generation vs DA")
    plt.scatter(df_avg['alpha'], df_avg['DiffHeur'], label = "Heuristic vs DA")

    plt.xlabel("alpha")
    plt.legend()
    plt.ylabel('Improvement in expected rank')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
    name_title = 'Improvement in expected rank vs DA (beta = ' + str(beta_in) + ')'
    plt.title(name_title)

    name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_percent_beta_" + str(beta_in) + '.pdf'
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")

def AvgRankImpr_absolute(S_vector: list[SolutionReport], name: str, beta_in:float, print_out = False):
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

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]
    plt.figure(figsize=(5,4))
    plt.scatter(df_avg['alpha'], df_avg['DiffEE'], label = "Erdil & Ergin vs DA")
    plt.scatter(df_avg['alpha'], df_avg['DiffCG'], label = "Column Generation vs DA")
    plt.scatter(df_avg['alpha'], df_avg['DiffHeur'], label = "Heuristic vs DA")

    plt.xlabel("alpha")
    plt.legend()
    plt.ylabel('Improvement in expected rank')
    name_title = 'Improvement in expected rank vs DA (beta = ' + str(beta_in) + ')'
    plt.title(name_title)

    name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_abs_beta_" + str(beta_in) + '.pdf'
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")


def AvgIndRankImpr(S_vector: list[SolutionReport], name: str, beta_in:float, print_out = False):
    
    
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

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]
    plt.figure(figsize=(5,4))
    plt.scatter(df_avg['alpha'], df_avg['average_rank_increase_US'], label = "Column generation")
    plt.scatter(df_avg['alpha'], df_avg['average_rank_increase_EE'], label = "Erdil & Ergin")

    plt.xlabel("alpha")
    plt.legend()
    plt.ylabel('Average individual increase in expected rank')
    name_title = 'Average individual increase in expected rank vs DA (beta = ' + str(beta_in) + ')'
    plt.title(name_title)

    name_plot = "Simulation Results/Plots/" + name + "/AvgIndRankImpr_abs_beta_" + str(beta_in) + '.pdf'
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")
