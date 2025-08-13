import numpy as np
import helper
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set up the plot style
plt.style.use('PaperDoubleFig.mplstyle.txt')
sns.set_style("white")
sns.set_palette(sns.color_palette("viridis")[::-1])

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


# Distributions to make plots for
# distr_list = [helper.normal_params, helper.exp_poisson_params, helper.time_varying_normal_params, helper.multi_params]
distr_list = [helper.multi_params]


for distr in distr_list:
    file_name = f'./data/data_{distr["budget_name"]}_{distr["demand_name"]}.csv'

    print(f'Reading data from: {file_name}')
    # Load data
    df = pd.read_csv(file_name)

    print(df.head(5))


    # Pivot the DataFrame so each metric becomes a column
    df_pivot = df.pivot_table(index=['delta', 'b', 'h', 'M', 'budget_name', 'mu_b', 'sigma_b', 'demand_name', 'mu_n', 'sigma_n'],
                            columns='metric', values='value', aggfunc='mean').reset_index()


    # Generate scatter plot for Efficiency vs M, grouped by Algorithm and unique parameter combinations
    unique_combinations = df_pivot[['sigma_n', 'sigma_b', 'mu_n', 'mu_b', 'demand_name', 'budget_name']].drop_duplicates()

    for _, params in unique_combinations.iterrows(): # unique combinations of parameters for the experiments
        subset = df_pivot[
            (df_pivot['sigma_n'] == params['sigma_n']) &
            (df_pivot['sigma_b'] == params['sigma_b']) &
            (df_pivot['mu_n'] == params['mu_n']) &
            (df_pivot['mu_b'] == params['mu_b']) &
            (df_pivot['demand_name'] == params['demand_name']) &
            (df_pivot['budget_name'] == params['budget_name'])
        ]

        subset = subset[subset["delta"] != np.inf]


        # print(f'Unique m values: {subset["M"].unique()}')
        midpoint = int(len(subset["M"].unique()) / 2)
        M_to_plot = [subset["M"].unique()[0], subset["M"].unique()[midpoint], subset["M"].unique()[-1]]

        print(f"M values to plot: {M_to_plot}")
        subset = subset[subset["M"].isin(M_to_plot)]

        # Get unique M values
        unique_M = np.sort(subset['M'].unique())
        num_m = len(unique_M)

        # Create a discrete color palette with equal spacing
        palette = sns.color_palette("viridis", n_colors=num_m)[::-1]

        # Assign colors to each delta value
        M_color_map = {M: palette[i] for i, M in enumerate(unique_M)}


        plt.figure(figsize=(12, 8))
    #     # Use the discrete color mapping for the `hue` parameter
        # plt.scatterplot
        # Create a line plot where each line corresponds to a unique value of M
        sns.lineplot(
            data=subset.sort_values(by=['M', 'fair']),  # Sort by M and fairness for smooth lines
            x='fair',
            y='efficiency',
            hue='M',
            marker="o",
            palette=M_color_map
        )
    #     sns.lineplot(data=subset, x='M', y='efficiency', hue='delta', palette=delta_color_map)
        
        plt.xlabel(r'$\Delta_{\mathrm{Fair}}$')
        plt.ylabel(r'$\Delta_{\mathrm{Efficiency}}$')
        plt.legend().remove() # This will remove the legend from the plot
        
                # Create a colorbar legend
        norm = mpl.colors.Normalize(vmin=min(unique_M), vmax=max(unique_M))
        sm = mpl.cm.ScalarMappable(cmap="viridis_r", norm=norm)
        sm.set_array([])  # Needed for the colorbar
        cbar = plt.colorbar(sm)
        cbar.set_label(r'$M$')  # Label for the colorbar

        # Override colorbar ticks: set lowest value to "0" and highest to "M"
        cbar.set_ticks([10, 100])
        cbar.set_ticklabels([r'$10$', rf'$100$'])

    #     # Save the plot
        filename = f"./figures/{distr['budget_name']}_{distr['demand_name']}_tradeoff_{params['sigma_n']}_{params['sigma_b']}_{params['mu_n']}_{params['mu_b']}.pdf"
        plt.savefig(filename, bbox_inches='tight')
    #     # plt.show()
        plt.close('all')
