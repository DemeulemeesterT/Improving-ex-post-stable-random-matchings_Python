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

def plot_avg_rank_alpha_beta_final(file_name: str, print_out = False):
    # Makes plots that show the fraction of improving students and their improvements
    # as a function of alpha matters for different betas, and different student numbers.

    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)

    #df_avg = create_average_df(df, True)

    # Help of ChatGPT to create plots

    # Columns to plot
    y_cols = [
        "avg_rank_impr_EE_DA",
        "1_avg_rank_impr_DA"#,
        #"2_avg_rank_impr_DA"
    ]

    group_cols = ["n_stud", "beta", "alpha"]

    # =========================
    # Aggregate: mean & std over seeds
    # =========================

    agg = (
        df[group_cols + y_cols]
        .groupby(group_cols)
        .agg(["mean", "std"])
    )

    # Flatten column names
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    # =========================
    # Determine common y-axis limit
    # =========================

    upper_cols = [f"{c}_mean" for c in y_cols] + [f"{c}_std" for c in y_cols]
    y_max = (
        agg[[f"{c}_mean" for c in y_cols]]
        .add(agg[[f"{c}_std" for c in y_cols]].values)
        .max()
        .max()
    )

    y_lim_top = 1.05 * y_max

    # =========================
    # Plot setup
    # =========================

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharey=True)

    # Template-style colors & linestyles (B/W friendly)
    styles = {
        "avg_rank_impr_EE_DA": dict(
            color="slateblue",
            linestyle="dashed",
            linewidth=1.2,
            label="EE"
        ),
        "1_avg_rank_impr_DA": dict(
            color="orangered",
            linestyle="solid",
            linewidth=1.2,
            label="SD-DA-CG"
        ),
        "2_avg_rank_impr_DA": dict(
            color="goldenrod",
            linestyle=(5, (10, 3)),
            linewidth=1.2,
            label="Method 2 vs DA"
        ),
    }

    # Panel definitions: (row, col, n_stud, beta)
    panels = [
        (0, 0, 40, 0.2),
        (0, 1, 40, 0.6),
        (1, 0, 80, 0.2),
        (1, 1, 80, 0.6),
    ]

    # =========================
    # Plot panels
    # =========================

    for r, c, n_stud, beta in panels:
        ax_rc = ax[r][c]

        sub = agg[(agg.n_stud == n_stud) & (agg.beta == beta)]
        sub = sub.sort_values("alpha")

        for col in y_cols:
            # Mean line
            ax_rc.plot(
                sub.alpha,
                sub[f"{col}_mean"],
                **styles[col]
            )

            # ±1 standard deviation band
            # Colored error bands
            ax_rc.fill_between(
                sub.alpha,
                sub[f"{col}_mean"] - sub[f"{col}_std"],
                sub[f"{col}_mean"] + sub[f"{col}_std"],
                color=styles[col]["color"],
                alpha=0.1,
                linewidth=0
            )

            #ax_rc.errorbar(
            #    sub.alpha,
            #    sub[f"{col}_mean"],
            #    yerr=sub[f"{col}_std"],
            #    color=styles[col]["color"],
            #    linestyle=styles[col]["linestyle"],
            #    linewidth=styles[col]["linewidth"],
            #    capsize=3,
            #    elinewidth=0.8,
            #    fmt='none'
            #)

        ax_rc.set_title(rf"$n={n_stud},\ \beta={beta}$")
        ax_rc.set_xlabel(r"$\alpha$")
        ax_rc.grid(axis="y", linewidth=0.5, color="lightgrey")
        ax_rc.set_ylim(bottom=0, top=y_lim_top)

    # Y-label only on left column
    ax[0][0].set_ylabel("Avg. rank improvement")
    ax[1][0].set_ylabel("Avg. rank improvement")

    # =========================
    # Shared legend (template-style)
    # =========================

    handles = [
        ax[0][0].lines[0],
        ax[0][0].lines[1]#,
        #ax[0][0].lines[2],
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.06),
        loc="lower center",
        ncol=3
    )

    plt.tight_layout()
    name_plot = "Simulation Results/Plots/" + file_name + "/AvgRankImpr_absolute.pdf"
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")
    plt.show()




def plot_fraction_impr_alpha_beta_final(file_name: str, print_out = False):
    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)

    # =========================
    # Create fraction columns
    # =========================

    df["frac_impr_EE_DA"] = df["n_stud_impr_EE_DA"] / df["n_stud"]
    df["frac_impr_1_DA"]  = df["1_n_stud_impr_DA"] / df["n_stud"]

    y_cols = [
        "frac_impr_EE_DA",
        "frac_impr_1_DA"
    ]

    group_cols = ["n_stud", "beta", "alpha"]

    # =========================
    # Aggregate: mean & std over seeds
    # =========================

    agg = (
        df[group_cols + y_cols]
        .groupby(group_cols)
        .agg(["mean", "std"])
    )

    # Flatten column names
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    # =========================
    # Determine common y-axis limit
    # =========================

    y_max = (
        agg[[f"{c}_mean" for c in y_cols]]
        .add(agg[[f"{c}_std" for c in y_cols]].values)
        .max()
        .max()
    )

    y_lim_top = min(1.05 * y_max, 1.05)  # fractions should not exceed 1 much

    # =========================
    # Plot setup
    # =========================

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharey=True)

    # Template-style colors & linestyles
    styles = {
        "frac_impr_EE_DA": dict(
            color="slateblue",
            linestyle="dashed",
            linewidth=1.2,
            label="EE"
        ),
        "frac_impr_1_DA": dict(
            color="orangered",
            #linestyle="dashed",
            linewidth=1.2,
            label="SD-DA-CG"
        ),
    }

    # Panel definitions: (row, col, n_stud, beta)
    panels = [
        (0, 0, 40, 0.2),
        (0, 1, 40, 0.6),
        (1, 0, 80, 0.2),
        (1, 1, 80, 0.6),
    ]

    # =========================
    # Plot panels
    # =========================

    for r, c, n_stud, beta in panels:
        ax_rc = ax[r][c]

        sub = agg[(agg.n_stud == n_stud) & (agg.beta == beta)]
        sub = sub.sort_values("alpha")

        for col in y_cols:
            # Mean line
            ax_rc.plot(
                sub.alpha,
                sub[f"{col}_mean"],
                **styles[col]
            )

            # ±1 standard deviation band
            ax_rc.fill_between(
                sub.alpha,
                sub[f"{col}_mean"] - sub[f"{col}_std"],
                sub[f"{col}_mean"] + sub[f"{col}_std"],
                color=styles[col]["color"],
                alpha=0.1,
                linewidth=0
            )

        ax_rc.set_title(rf"$n={n_stud},\ \beta={beta}$")
        ax_rc.set_xlabel(r"$\alpha$")
        ax_rc.grid(axis="y", linewidth=0.5, color="lightgrey")
        ax_rc.set_ylim(bottom=0, top=y_lim_top)
        ax_rc.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # Y-label only on left column
    ax[0][0].set_ylabel("Fraction of improving students")
    ax[1][0].set_ylabel("Fraction of improving students")

    # =========================
    # Shared legend
    # =========================

    handles = [
        ax[0][0].lines[0],
        ax[0][0].lines[1],
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.06),
        loc="lower center",
        ncol=2
    )

    plt.tight_layout()
    name_plot = "Simulation Results/Plots/" + file_name + "/Fraction_Impr_stud.pdf"
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")
    plt.show()




def plot_avg_rank_CG_eval_alpha_beta_final(file_name: str, print_out = False):
    # Makes plots that show the fraction of improving students and their improvements
    # as a function of alpha matters for different betas, and different student numbers.

    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)

    #df_avg = create_average_df(df, True)

    # Help of ChatGPT to create plots

    # =========================
    # Create improvement column heuristic
    # =========================

    df["1_avg_rank_impr_heur_DA"] = df["avg_rank_DA"] - df["1_avg_rank_heur"] 

    # Columns to plot
    y_cols = [
        "1_avg_rank_impr_DA",
        "4_avg_rank_impr_DA",
        "1_avg_rank_impr_heur_DA"
    ]

    group_cols = ["n_stud", "beta", "alpha"]

    # =========================
    # Aggregate: mean & std over seeds
    # =========================

    agg = (
        df[group_cols + y_cols]
        .groupby(group_cols)
        .agg(["mean", "std"])
    )

    # Flatten column names
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    #print(agg)

    # =========================
    # Determine common y-axis limit
    # =========================

    upper_cols = [f"{c}_mean" for c in y_cols] + [f"{c}_std" for c in y_cols]
    y_max = (
        agg[[f"{c}_mean" for c in y_cols]]
        #.add(agg[[f"{c}_std" for c in y_cols]].values)
        .max()
        .max()
    )

    y_lim_top = 1.05 * y_max

    # =========================
    # Plot setup
    # =========================

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharey=True)

    # Template-style colors & linestyles (B/W friendly)
    styles = {
        "1_avg_rank_impr_heur_DA": dict(
            color="forestgreen",
            linestyle="dashed",
            linewidth=1.2,
            label="SD-DA-heur"
        ),
        "1_avg_rank_impr_DA": dict(
            color="orangered",
            linestyle="solid",
            linewidth=1.2,
            label="SD-DA-CG"
        ),
        "4_avg_rank_impr_DA": dict(
            color="goldenrod",
            linestyle=(5, (10, 3)),
            linewidth=1.2,
            label="SD-DA-SAMPLE-10000"
        ),
    }

    # Panel definitions: (row, col, n_stud, beta)
    panels = [
        (0, 0, 40, 0.2),
        (0, 1, 40, 0.6),
        (1, 0, 80, 0.2),
        (1, 1, 80, 0.6),
    ]

    # =========================
    # Plot panels
    # =========================

    for r, c, n_stud, beta in panels:
        ax_rc = ax[r][c]

        sub = agg[(agg.n_stud == n_stud) & (agg.beta == beta)]
        sub = sub.sort_values("alpha")

        for col in y_cols:
            # Mean line
            ax_rc.plot(
                sub.alpha,
                sub[f"{col}_mean"],
                **styles[col]
            )

            # ±1 standard deviation band
            # Colored error bands
            #ax_rc.fill_between(
            #    sub.alpha,
            #    sub[f"{col}_mean"] - sub[f"{col}_std"],
            #    sub[f"{col}_mean"] + sub[f"{col}_std"],
            #    color=styles[col]["color"],
            #    alpha=0.1,
            #    linewidth=0
            #)

            #ax_rc.errorbar(
            #    sub.alpha,
            #    sub[f"{col}_mean"],
            #    yerr=sub[f"{col}_std"],
            #    color=styles[col]["color"],
            #    linestyle=styles[col]["linestyle"],
            #    linewidth=styles[col]["linewidth"],
            #    capsize=3,
            #    elinewidth=0.8,
            #    fmt='none'
            #)

        ax_rc.set_title(rf"$n={n_stud},\ \beta={beta}$")
        ax_rc.set_xlabel(r"$\alpha$")
        ax_rc.grid(axis="y", linewidth=0.5, color="lightgrey")
        ax_rc.set_ylim(bottom=0, top=y_lim_top)

    # Y-label only on left column
    ax[0][0].set_ylabel("Avg. rank improvement")
    ax[1][0].set_ylabel("Avg. rank improvement")

    # =========================
    # Shared legend (template-style)
    # =========================

    handles = [
        ax[0][0].lines[0],
        ax[0][0].lines[1],
        ax[0][0].lines[2],
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.06),
        loc="lower center",
        ncol=3
    )

    plt.tight_layout()
    name_plot = "Simulation Results/Plots/" + file_name + "/AvgRankImpr_CG_eval_absolute.pdf"
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")
    plt.show()


def evaluate_CG(file_name: str, print_out = False):
    # Create directory if it doesn't exist
    name_folder = 'Simulation Results/Plots/' + file_name
    os.makedirs(name_folder, exist_ok=True)

    # Read in csv data file
    csv_file_path = "Simulation Results/" + file_name + ".csv"
    df = pd.read_csv(csv_file_path)

    # Set improvement by CG as zero. State how much better/worse heuristic and sampling method performed.
    df["1_avg_rank_impr_heur_DA"] = df["avg_rank_DA"] - df["1_avg_rank_heur"] 

    df["1_heur_vs_DA"] = df["1_avg_rank_impr_heur_DA"] - df["1_avg_rank_impr_DA"]
    df["4_sample_vs_DA"] = df["4_avg_rank_impr_DA"] - df["1_avg_rank_impr_DA"]
    df["1_rescaled_avg_rank_impr_DA"] = 0 # just reference point

    # Columns to plot
    y_cols = [
        "1_heur_vs_DA",
        "4_sample_vs_DA",
        "1_rescaled_avg_rank_impr_DA"
    ]

    group_cols = ["n_stud", "beta", "alpha"]

    # =========================
    # Aggregate: mean & std over seeds
    # =========================

    agg = (
        df[group_cols + y_cols]
        .groupby(group_cols)
        .agg(["mean", "std"])
    )

    # Flatten column names
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    print(agg)

        # =========================
    # Determine common y-axis limit
    # =========================

    upper_cols = [f"{c}_mean" for c in y_cols] + [f"{c}_std" for c in y_cols]
    y_max = (
        agg[[f"{c}_mean" for c in y_cols]]
        .add(agg[[f"{c}_std" for c in y_cols]].values)
        .max()
        .max()
    )

    y_min = (
        agg[[f"{c}_mean" for c in y_cols]]
        .add(-agg[[f"{c}_std" for c in y_cols]].values)
        .min()
        .min()
    )

    y_lim_top = 1.05 * y_max
    y_lim_bottom = 1.05 * y_min

    # =========================
    # Plot setup
    # =========================

    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharey=True)

    # Template-style colors & linestyles (B/W friendly)
    styles = {
        "1_heur_vs_DA": dict(
            color="forestgreen",
            linestyle="dashed",
            linewidth=1.2,
            label="SD-DA-heur"
        ),
        "1_rescaled_avg_rank_impr_DA": dict(
            color="orangered",
            linestyle="solid",
            linewidth=1.2,
            label="SD-DA-CG"
        ),
        "4_sample_vs_DA": dict(
            color="goldenrod",
            linestyle=(5, (10, 3)),
            linewidth=1.2,
            label="SD-DA-SAMPLE-10000"
        ),
    }

    # Panel definitions: (row, col, n_stud, beta)
    panels = [
        (0, 0, 40, 0.2),
        (0, 1, 40, 0.6),
        (1, 0, 80, 0.2),
        (1, 1, 80, 0.6),
    ]

    # =========================
    # Plot panels
    # =========================

    for r, c, n_stud, beta in panels:
        ax_rc = ax[r][c]

        sub = agg[(agg.n_stud == n_stud) & (agg.beta == beta)]
        sub = sub.sort_values("alpha")

        for col in y_cols:
            # Mean line
            ax_rc.plot(
                sub.alpha,
                sub[f"{col}_mean"],
                **styles[col]
            )

            # ±1 standard deviation band
            # Colored error bands
            ax_rc.fill_between(
                sub.alpha,
                sub[f"{col}_mean"] - sub[f"{col}_std"],
                sub[f"{col}_mean"] + sub[f"{col}_std"],
                color=styles[col]["color"],
                alpha=0.2,
                linewidth=0
            )

            #ax_rc.errorbar(
            #    sub.alpha,
            #    sub[f"{col}_mean"],
            #    yerr=sub[f"{col}_std"],
            #    color=styles[col]["color"],
            #    linestyle=styles[col]["linestyle"],
            #    linewidth=styles[col]["linewidth"],
            #    capsize=3,
            #    elinewidth=0.8,
            #    fmt='none'
            #)

        ax_rc.set_title(rf"$n={n_stud},\ \beta={beta}$")
        ax_rc.set_xlabel(r"$\alpha$")
        ax_rc.grid(axis="y", linewidth=0.5, color="lightgrey")
        ax_rc.set_ylim(bottom=y_lim_bottom, top=y_lim_top)

    # Y-label only on left column
    ax[0][0].set_ylabel("Avg. rank improvement vs. SD-DA-CG")
    ax[1][0].set_ylabel("Avg. rank improvement vs. SD-DA-CG")

    # =========================
    # Shared legend (template-style)
    # =========================

    handles = [
        ax[0][0].lines[0],
        ax[0][0].lines[1],
        ax[0][0].lines[2],
    ]

    fig.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.06),
        loc="lower center",
        ncol=3
    )

    plt.tight_layout()
    name_plot = "Simulation Results/Plots/" + file_name + "/AvgRankImpr_CG_eval_relative.pdf"
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")
    plt.show()





