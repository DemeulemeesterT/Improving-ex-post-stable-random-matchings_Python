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

def SimulationsEvaluate_alpha_beta(file_name: str, print_out = False):
    # Makes plots that show the fraction of improving students and their improvements
    # as a function of alpha matters for different betas, and different student numbers.

    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)


    # Average overall improvement in rank
    beta_in_vect = [0.2, 0.6]
    for beta in beta_in_vect:
        # RECENT FUNCTIONS
        # Fraction of improving students
        Fraction_improving_students_filter(df, file_name, beta, print_out)

        # Improvement in rank for improving students
        AvgRankImpr_absolute_filter(df, file_name, beta, print_out)

        # Fraction of improving students for EADA
        # Fraction_improving_students_EADA(df, file_name, beta, print_out)

        # OLDER FUNCTIONS

        # AvgRankImpr_percent_filter(df, file_name, beta, print_out)
        #AvgRankImpr_percent(df, file_name, beta, print_out) # With EADA
        #AvgRankImpr_absolute(df, file_name, beta_in, print_out)

        # Fraction of improving students
        # Fraction_improving_students_filter(df, file_name, beta, print_out)

        # Average individual improvement in rank (among improving agents)
        #AvgIndRankImpr(df, file_name, beta, print_out)

        # Display how the improvement differs with respect to n_stud
        #AvgImprovByNStud(df, file_name, beta_in, print_out)

        # Display by number of schools, one graph for each value of n_stud
        #AvgImprovByNSschools(df, file_name, beta_in, print_out)

def SimulationsEvaluate_EADA(file_name: str, print_out = False):
    # Makes plots that show the fraction of improving students and their improvements for EADA
    # as a function of alpha matters for different betas, and different student numbers.
    
    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)


    # Average overall improvement in rank
    beta_in_vect = [0.2, 0.6]
    for beta in beta_in_vect:
        # Fraction of improving students for EADA
        Fraction_improving_students_EADA(df, file_name, beta, print_out)

        # Improvement in rank for improving students
        AvgRankImpr_absolute_EADA(df, file_name, beta, print_out)


def SimulationsEvaluate_alpha_beta(file_name: str, print_out = False):
    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)


    # Average overall improvement in rank
    beta_in_vect = [0.2, 0.6]
    for beta in beta_in_vect:
        # RECENT FUNCTIONS
        # Fraction of improving students
        Fraction_improving_students_filter(df, file_name, beta, print_out)

        # Improvement in rank for improving students
        AvgRankImpr_absolute_filter(df, file_name, beta, print_out)

        # Fraction of improving students for EADA
        Fraction_improving_students_EADA(df, file_name, beta, print_out)



def AvgRankImpr_percent(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    #print(df[df['beta']==1])

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods+1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            legend.append("LP-heur EADA")    
    
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    df['DiffEE'] = (df['avg_rank_DA'] - df['avg_rank_EE'])/df['avg_rank_DA'] # Difference in average rank Erdil & Ergin vs DA
    df['DiffEADA'] = (df['avg_rank_DA'] - df['avg_rank_EADA'])/df['avg_rank_DA'] # Difference in average rank EADA vs DA 

    counter = 1
    for s in labels:
        # Create mask for rows where both values are not nan
        mask = df[['avg_rank_DA', f'{counter}_avg_rank_heur']].notna().all(axis=1)
        df[f'DiffHeur{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffHeur{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_heur']
        ) / df.loc[mask, 'avg_rank_DA']


        df[f'DiffResult{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffResult{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_result']
        ) / df.loc[mask, 'avg_rank_DA']

        counter = counter + 1
    
    print(df[['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "avg_rank_DA", "avg_rank_EADA", "3_avg_rank_heur"]])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    # Add column to count how many times EADA couldn't be sd-dominated for the parameters
    counter =1 
    for s in labels:
        if s == "SD_UPON_EADA":
            nan_counts = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta'])[f'{counter}_avg_rank_heur']\
               .apply(lambda x: x.isna().sum())\
               .reset_index(name='n_nans_sd_dom_EADA')
            
            df_avg = df_avg.merge(nan_counts, on=['n_stud', 'n_schools', 'alpha', 'beta'])
        counter = counter + 1

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['DiffResult1'].max()

    print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "3_avg_rank_heur", "n_nans_sd_dom_EADA", "3_n_stud_impr_EADA", "3_avg_rank_impr_EADA"]])

    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(5,4))
        plt.plot(df_n['alpha'], df_n['DiffEE'], label = "Erdil & Ergin")
        plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}'], label = labels[counter - 1] + ' (heuristic)')
            # Check if better result column generation
            df_n['differs'] = (df_n[f'DiffHeur{counter}'] - df_n[f'DiffResult{counter}']).abs() >= 0.001
            if df_n['differs'].any == True:
                 plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
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

def AvgRankImpr_percent_filter(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # More control on which results are shown
    #print(df[df['beta']==1])

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods+1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        #elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            #legend.append("LP-heur EADA")    
    
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    df['DiffEE'] = (df['avg_rank_DA'] - df['avg_rank_EE'])/df['avg_rank_DA'] # Difference in average rank Erdil & Ergin vs DA
    #df['DiffEADA'] = (df['avg_rank_DA'] - df['avg_rank_EADA'])/df['avg_rank_DA'] # Difference in average rank EADA vs DA 

    counter = 1
    for s in labels:
        # Create mask for rows where both values are not nan
        mask = df[['avg_rank_DA', f'{counter}_avg_rank_heur']].notna().all(axis=1)
        df[f'DiffHeur{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffHeur{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_heur']
        ) / df.loc[mask, 'avg_rank_DA']


        df[f'DiffResult{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffResult{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_result']
        ) / df.loc[mask, 'avg_rank_DA']

        counter = counter + 1
    
    print(df[['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "avg_rank_DA", "avg_rank_EADA", "3_avg_rank_heur"]])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    # Add column to count how many times EADA couldn't be sd-dominated for the parameters
    counter =1 
    for s in labels:
        #if s == "SD_UPON_EADA":
        #    nan_counts = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta'])[f'{counter}_avg_rank_heur']\
        #       .apply(lambda x: x.isna().sum())\
        #       .reset_index(name='n_nans_sd_dom_EADA')
            
        #    df_avg = df_avg.merge(nan_counts, on=['n_stud', 'n_schools', 'alpha', 'beta'])
        counter = counter + 1

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['DiffResult1'].max()

    #print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "3_avg_rank_heur", "n_nans_sd_dom_EADA", "3_n_stud_impr_EADA", "3_avg_rank_impr_EADA"]])

    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            if s != "SD_UPON_EADA":
                plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}'], label = labels[counter - 1] + ' (heuristic)')
                # Check if better result column generation
                df_n['differs'] = (df_n[f'DiffHeur{counter}'] - df_n[f'DiffResult{counter}']).abs() >= 0.001
                if df_n['differs'].any == True:
                    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
            counter = counter  +1

        plt.plot(df_n['alpha'], df_n['DiffEE'], label = "Erdil & Ergin")

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Improvement in expected rank')

        plt.ylim(-0.005, max_diff + 0.005)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Improvement in expected rank vs DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_percent_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")


def AvgRankImpr_absolute_filter(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Function that plots the number of positions improved in ranking (no EADA)

    # More control on which results are shown
    #print(df[df['beta']==1])

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods + 1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        #elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            #legend.append("LP-heur EADA")    
    
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    df['DiffEE'] = (df['avg_rank_impr_EE_DA']) # Difference in average rank Erdil & Ergin vs DA
    #df['DiffEADA'] = (df['avg_rank_DA'] - df['avg_rank_EADA'])/df['avg_rank_DA'] # Difference in average rank EADA vs DA 

    counter = 1
    print(labels, "labels")
    for s in labels:
        print("s", s)
        # Create mask for rows where both values are not nan
        mask = df[['avg_rank_DA', f'{counter}_avg_rank_heur']].notna().all(axis=1)
        df[f'DiffHeur{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffHeur{counter}'] = (
            df.loc[mask, f'{counter}_avg_rank_impr_DA']
        )


        df[f'DiffResult{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffResult{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_result']
        ) 

        counter = counter + 1
    
    #print(df[['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "avg_rank_DA", "avg_rank_EADA", "3_avg_rank_heur"]])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    # Add column to count how many times EADA couldn't be sd-dominated for the parameters
    counter =1 
    for s in labels:
        #if s == "SD_UPON_EADA":
        #    nan_counts = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta'])[f'{counter}_avg_rank_heur']\
        #       .apply(lambda x: x.isna().sum())\
        #       .reset_index(name='n_nans_sd_dom_EADA')
            
        #    df_avg = df_avg.merge(nan_counts, on=['n_stud', 'n_schools', 'alpha', 'beta'])
        counter = counter + 1

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['DiffHeur1'].max()

    #print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "3_avg_rank_heur", "n_nans_sd_dom_EADA", "3_n_stud_impr_EADA", "3_avg_rank_impr_EADA"]])

    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            if s != "SD_UPON_EADA":
                plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}'], label = str(labels[counter - 1]) + ' (heuristic)')
                # Check if better result column generation
                df_n['differs'] = (df_n[f'DiffHeur{counter}'] - df_n[f'DiffResult{counter}']).abs() >= 0.001
                if df_n['differs'].any == True:
                    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = str(labels[counter - 1]) + ' (CG)')
            counter = counter  +1
        
        plt.plot(df_n['alpha'], df_n['DiffEE'], label = "DA + SIC")

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Improvement in expected rank vs DA')

        plt.ylim(-0.005, max_diff + 0.05)
        #plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Average improvement in rank for improving students vs DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_absolute_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")



def AvgRankImpr_absolute_EADA(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Function that plots the number of positions improved in ranking (including EADA)

    # More control on which results are shown
    #print(df[df['beta']==1])

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods + 1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        #elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            #legend.append("LP-heur EADA")    

    # Find which instanced could sd-improve upon EADA
    for i in range(1, n_sol_methods+1):
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            # Create mask for when EADA could be sd-improved upon (i.e., where both values are not nan)
            mask = df[['avg_rank_DA', f'{i}_avg_rank_heur']].notna().all(axis=1)

    counter = 1
    for s in labels:
        df.loc[mask, f'DiffHeur{counter}_DA'] = (
            df.loc[mask, f'{counter}_avg_rank_impr_DA'])
        counter = counter + 1


    # Also compute improvement of improving students for EADA and EE, 
    # but only for instances where EADA could be sd-doninated upon!
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    df.loc[mask, 'DiffEE_DA'] = (df.loc[mask, 'avg_rank_impr_EE_DA']) 
    df.loc[mask,'DiffEADA_DA'] = (df.loc[mask, 'avg_rank_impr_EADA_DA'])


    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    # Add column to count how many times EADA couldn't be sd-dominated for the parameters
    counter =1 

    for s in labels:
        if s == "SD_UPON_EADA":
            # Count number of nans (helped by ChatGPT)
            sd_upon_EADA_count = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta'])[f'{counter}_avg_rank_heur']\
               .apply(lambda x: x.notna().sum())\
               .reset_index(name='n_sd_upon_EADA_count')
            
            df_avg = df_avg.merge(sd_upon_EADA_count , on=['n_stud', 'n_schools', 'alpha', 'beta'])
        counter = counter + 1

    df_avg = df_avg[df_avg['beta'] == beta_in]
    print("df_avg for rank improvement")
    print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1_DA', 'DiffHeur2_DA', "DiffHeur3_DA", "3_avg_rank_heur", "n_sd_upon_EADA_count", "DiffEADA_DA"]])

    max_diff_DA = df_avg['DiffHeur1_DA'].max()
    max_diff_EE = df_avg['DiffHeur1_EE'].max()
    max_diff_EADA = df_avg['DiffHeur3_DA'].max()

    max_diff = max(max_diff_DA, max_diff_EE, max_diff_EADA)


    # Diff wrt DA
    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        
        #sd_count percentage
        df_n['sd_upon_EADA_percentage'] = df_n['n_sd_upon_EADA_count']/10

        print(df_n[['alpha', 'sd_upon_EADA_percentage']])

        # Histogram with number of times we could improve upon EADA
        plt.bar(df_n['alpha'], df_n['sd_upon_EADA_percentage'], width = 0.08, alpha = 0.4)  
    


        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}_DA'], label = labels[counter - 1] + ' (heuristic)', marker = ".")
            # Check if better result column generation
            # df_n['differs'] = (df_n[f'DiffHeur{counter}_DA'] - df_n[f'DiffResult{counter}_DA']).abs() >= 0.001
            # if df_n['differs'].any == True:
            #    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
            counter = counter  +1

        plt.plot(df_n['alpha'], df_n['DiffEE_DA'], label = "DA + SIC", marker = ".")

        plt.plot(df_n['alpha'], df_n['DiffEADA_DA'], label = "EADA", marker = ".")

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Improvement in expected rank vs DA')

        plt.ylim(-0.005, max_diff + 0.05)
        #plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Average improvement in rank for improving students vs DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/AvgRankImpr_absolute_(withEADA)_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")






def Fraction_improving_students_filter(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Fraction of improving students visualised (no EADAM)
    # Do this once upon DA, and once upon EE

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods+1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        #elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            #legend.append("LP-heur EADA")    
    
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    print(df['n_stud_impr_EE_DA'])
    df['DiffEE_DA'] = (df['n_stud_impr_EE_DA'])/df['n_stud'] # Fraction of students improving in EE vs DA
    #df['DiffEADA'] = (df['avg_rank_DA'] - df['avg_rank_EADA'])/df['avg_rank_DA'] # Difference in average rank EADA vs DA 

    counter = 1
    for s in labels:
        # Create mask for rows where both values are not nan
        mask = df[['avg_rank_DA', f'{counter}_avg_rank_heur']].notna().all(axis=1)
        df[f'DiffHeur{counter}_DA'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffHeur{counter}_DA'] = (
            df.loc[mask, f'{counter}_n_stud_impr_DA']
        ) / df.loc[mask, 'n_stud']

        df.loc[mask, f'DiffHeur{counter}_EE'] = (
            df.loc[mask, f'{counter}_n_stud_impr_EE']
        ) / df.loc[mask, 'n_stud']
        counter = counter + 1
    
    print(df[['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'DiffHeur1_DA', 'DiffHeur2_DA', "DiffHeur3_DA", "DiffEE_DA"]])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns


    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff_DA = df_avg['DiffHeur1_DA'].max()
    max_diff_EE = df_avg['DiffHeur1_EE'].max()

    print("max_diff_DA", max_diff_DA)
    print("max_diff_EE", max_diff_EE)

    #print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "3_avg_rank_heur", "n_nans_sd_dom_EADA", "3_n_stud_impr_EADA", "3_avg_rank_impr_EADA"]])


    # Diff wrt DA
    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            if s != "SD_UPON_EADA":
                plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}_DA'], label = str(labels[counter - 1]) + ' (heuristic)')
                # Check if better result column generation
                # df_n['differs'] = (df_n[f'DiffHeur{counter}_DA'] - df_n[f'DiffResult{counter}_DA']).abs() >= 0.001
                # if df_n['differs'].any == True:
                #    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
            counter = counter  +1

        plt.plot(df_n['alpha'], df_n['DiffEE_DA'], label = "DA + SIC")

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Fraction of improving students upon DA')

        plt.ylim(-0.005, 1 + 0.005)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Fraction of improving students upon DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/FracImprStud_DA_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")

    # Diff wrt EE
    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            if s != "SD_UPON_EADA":
                plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}_EE'], label = str(labels[counter - 1]) + ' (heuristic)')
                # Check if better result column generation
                # df_n['differs'] = (df_n[f'DiffHeur{counter}_DA'] - df_n[f'DiffResult{counter}_DA']).abs() >= 0.001
                # if df_n['differs'].any == True:
                #    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
            counter = counter  +1

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Fraction of improving students upon EE')

        plt.ylim(-0.005, 1 + 0.005)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Fraction of improving students upon EE\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/FracImprStud_EE_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
        plt.savefig(name_plot, format="pdf", bbox_inches="tight")


def Fraction_improving_students_EADA(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Fraction of improving students visualised for EADA
    # Only show the instances for which we can sd-improve upon EADA
    # Plot improvement lines upon histogram that shows the percentage of instances in which we could
        # improve upon EADA
    # Do this once upon DA, and once upon EE

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods+1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            legend.append("LP-heur EADA")    
    
    
    # Find which instanced could sd-improve upon EADA
    for i in range(1, n_sol_methods+1):
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            # Create mask for when EADA could be sd-improved upon (i.e., where both values are not nan)
            mask = df[['avg_rank_DA', f'{i}_avg_rank_heur']].notna().all(axis=1)

    counter = 1
    for s in labels:
        df.loc[mask, f'DiffHeur{counter}_DA'] = (
            df.loc[mask, f'{counter}_n_stud_impr_DA']
        ) / df.loc[mask, 'n_stud']

        df.loc[mask, f'DiffHeur{counter}_EE'] = (
            df.loc[mask, f'{counter}_n_stud_impr_EE']
        ) / df.loc[mask, 'n_stud']
        counter = counter + 1

    # Also compute fraction of improving students for EADA and EE, 
    # but only for instances where EADA could be sd-doninated upon!
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    df.loc[mask, 'DiffEE_DA'] = (df.loc[mask,'n_stud_impr_EE_DA'])/df.loc[mask,'n_stud'] # Fraction of students improving in EE vs DA
    df.loc[mask,'DiffEADA_DA'] = (df.loc[mask,'n_stud_impr_EADA_DA'])/df.loc[mask,'n_stud'] # Fraction of students improving in EADA vs DA

    
    print(df[['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'DiffHeur1_DA', 'DiffHeur2_DA', "DiffHeur3_DA", "DiffEE_DA"]])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    # Add column to count how many times EADA couldn't be sd-dominated for the parameters
    counter =1 
    for s in labels:
        if s == "SD_UPON_EADA":
            # Count number of nans (helped by ChatGPT)
            sd_upon_EADA_count = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta'])[f'{counter}_avg_rank_heur']\
               .apply(lambda x: x.notna().sum())\
               .reset_index(name='n_sd_upon_EADA_count')
            
            df_avg = df_avg.merge(sd_upon_EADA_count , on=['n_stud', 'n_schools', 'alpha', 'beta'])
        counter = counter + 1

    print(df_avg)


    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff_DA = df_avg['DiffHeur1_DA'].max()
    max_diff_EE = df_avg['DiffHeur1_EE'].max()
    max_diff_EADA = df_avg['DiffHeur3_DA'].max()


    print("max_diff_DA", max_diff_DA)
    print("max_diff_EE", max_diff_EE)
    print("max_diff_EADA", max_diff_EADA)

    #print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "3_avg_rank_heur", "n_nans_sd_dom_EADA", "3_n_stud_impr_EADA", "3_avg_rank_impr_EADA"]])


    # Diff wrt DA
    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        
        #sd_count percentage
        df_n['sd_upon_EADA_percentage'] = df_n['n_sd_upon_EADA_count']/10

        print(df_n[['alpha', 'sd_upon_EADA_percentage']])

        # Histogram with number of times we could improve upon EADA
        plt.bar(df_n['alpha'], df_n['sd_upon_EADA_percentage'], width = 0.08, alpha = 0.4)  
    


        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}_DA'], label = str(labels[counter - 1]) + ' (heuristic)', marker = ".")
            # Check if better result column generation
            # df_n['differs'] = (df_n[f'DiffHeur{counter}_DA'] - df_n[f'DiffResult{counter}_DA']).abs() >= 0.001
            # if df_n['differs'].any == True:
            #    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
            counter = counter  +1

        plt.plot(df_n['alpha'], df_n['DiffEE_DA'], label = "DA + SIC", marker = ".")

        plt.plot(df_n['alpha'], df_n['DiffEADA_DA'], label = "EADA", marker = ".")

        plt.xlabel("alpha")
        plt.legend()
        plt.ylabel('Fraction of improving students upon DA')

        plt.ylim(-0.005, 1 + 0.08)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) # Express y-axis in percentages
        name_title = 'Fraction of improving students upon DA\n (n = ' + str(n_stud) + ', beta = ' + str(beta_in) + ')'
        plt.title(name_title)

        name_plot = "Simulation Results/Plots/" + name + "/FracImprStud_DA_(withEADA)_beta_" + str(n_stud) + '_' + str(beta_in) + '.pdf'
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


def AvgRankImpr_percent_filter(df: pd.DataFrame, name: str, beta_in:float, print_out = False):
    # Fraction of the students that are improving upon DA
    # Do not consider EADA in this function

    n_sol_methods = df["#_sol_methods"].iloc[0]
    labels = []
    legend = []
    for i in range(1, n_sol_methods+1):
        print('i',i, ' of ', n_sol_methods)
        labels.append(df[f'sol_{i}_label'].iloc[0])
        if df[f'sol_{i}_label'].iloc[0] == "SD_UPON_DA":
            legend.append("LP-heur DA")
        elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EE":
            legend.append("LP-heur EE")
        #elif df[f'sol_{i}_label'].iloc[0] == "SD_UPON_EADA":
            #legend.append("LP-heur EADA")    
    
    # Do pairwise comparisons already before averaging out (because nans might cause problems otherwise)
    # PERCENTAGE
    df['DiffEE'] = (df['avg_rank_DA'] - df['avg_rank_EE'])/df['avg_rank_DA'] # Difference in average rank Erdil & Ergin vs DA
    #df['DiffEADA'] = (df['avg_rank_DA'] - df['avg_rank_EADA'])/df['avg_rank_DA'] # Difference in average rank EADA vs DA 

    counter = 1
    for s in labels:
        # Create mask for rows where both values are not nan
        mask = df[['avg_rank_DA', f'{counter}_avg_rank_heur']].notna().all(axis=1)
        df[f'DiffHeur{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffHeur{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_heur']
        ) / df.loc[mask, 'avg_rank_DA']


        df[f'DiffResult{counter}'] = np.nan  # initialize with NaN
        
        df.loc[mask, f'DiffResult{counter}'] = (
            df.loc[mask, 'avg_rank_DA'] - df.loc[mask, f'{counter}_avg_rank_result']
        ) / df.loc[mask, 'avg_rank_DA']

        counter = counter + 1
    
    print(df[['n_stud', 'n_schools', 'alpha', 'beta', 'seed', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "avg_rank_DA", "avg_rank_EADA", "3_avg_rank_heur"]])

    # Average out
    df_avg = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta']).mean(numeric_only=True).reset_index()
        # 'numeric_only' enforces that non-numeric columns are not averaged (like labels)
        # Last command resets indices to keep using them as colummns

    # Add column to count how many times EADA couldn't be sd-dominated for the parameters
    counter =1 
    for s in labels:
        #if s == "SD_UPON_EADA":
        #    nan_counts = df.groupby(['n_stud', 'n_schools', 'alpha', 'beta'])[f'{counter}_avg_rank_heur']\
        #       .apply(lambda x: x.isna().sum())\
        #       .reset_index(name='n_nans_sd_dom_EADA')
            
        #    df_avg = df_avg.merge(nan_counts, on=['n_stud', 'n_schools', 'alpha', 'beta'])
        counter = counter + 1

    #print(df_avg)

    df_avg = df_avg[df_avg['beta'] == beta_in]

    max_diff = df_avg['DiffResult1'].max()

    #print(df_avg[['n_stud', 'n_schools', 'alpha', 'beta', 'DiffHeur1', 'DiffHeur2', "DiffHeur3", "3_avg_rank_heur", "n_nans_sd_dom_EADA", "3_n_stud_impr_EADA", "3_avg_rank_impr_EADA"]])

    for n_stud in df_avg['n_stud'].unique():
        df_n = df_avg[df_avg['n_stud'] == n_stud]
        plt.figure(figsize=(6,3))
        plt.plot(df_n['alpha'], df_n['DiffEE'], label = "Erdil & Ergin")
        #plt.plot(df_n['alpha'], df_n['DiffEADA'], label = "EADA")
        counter = 1
        for s in labels:
            if s != "SD_UPON_EADA":
                plt.plot(df_n['alpha'], df_n[f'DiffHeur{counter}'], label = labels[counter - 1] + ' (heuristic)')
                # Check if better result column generation
                df_n['differs'] = (df_n[f'DiffHeur{counter}'] - df_n[f'DiffResult{counter}']).abs() >= 0.001
                if df_n['differs'].any == True:
                    plt.plot(df_n['alpha'], df_n[f'DiffResult{counter}'], label = labels[counter - 1] + ' (CG)')
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

