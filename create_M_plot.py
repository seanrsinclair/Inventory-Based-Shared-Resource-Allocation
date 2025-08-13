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
distr_list = [helper.normal_params, helper.exp_poisson_params, helper.time_varying_normal_params, helper.multi_params]



for distr in distr_list:
    file_name = f'./data/data_{distr["budget_name"]}_{distr["demand_name"]}.csv'

    print(f'Reading data from: {file_name}')
    # Load data
    df = pd.read_csv(file_name)

    print(df.head(5))


    # Filter for rows where the metric is 'fair'
    fair_df = df[df['metric'] == 'fair']

    # Group by algorithm and compute the average of the fair metric
    avg_fair_per_algorithm = fair_df.groupby('algorithm')['value'].mean()

    # Get the maximum average fair metric for each algorithm
    max_avg_fair = avg_fair_per_algorithm.max()
    print(f'Max average fairness: {max_avg_fair}')



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

        # if distr['budget_name'] == 'exponential':
        # subset = subset[subset['delta'] < np.inf]


        # Replace efficiency values smaller than 10^-4 with 10^-4
        subset['efficiency'] = subset['efficiency'].apply(lambda x: max(x, 1e-4))

        # Get unique delta values
        non_inf_subset = subset[subset['delta'] != np.inf]

        unique_deltas = np.sort(non_inf_subset['delta'].unique())
        num_deltas = len(unique_deltas)
        print(f'Unique deltas: {unique_deltas}')

        # Create a discrete color palette with equal spacing
        palette = sns.color_palette("viridis", n_colors=num_deltas)[::-1]

        # Assign colors to each delta value
        delta_color_map = {delta: palette[i] for i, delta in enumerate(unique_deltas)}


        plt.figure(figsize=(12, 8))
        # Use the discrete color mapping for the `hue` parameter
        sns.lineplot(data=non_inf_subset, x='M', y='efficiency', hue='delta', palette=delta_color_map)


        # sns.lineplot(data = subset[subset['delta'] == np.inf], x = 'M', y = 'efficiency', color='r')
        
        plt.xlabel(r'$M$')
        plt.ylabel(r'$\log(\Delta_{\mathrm{Efficiency}})$')

        if distr['budget_name'] != 'normal_multi':
            plt.yscale('log')
            plt.ylim(1e-4, 1e1)
            plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
            [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])

        else:
            plt.yscale('log')
            plt.ylim(1e-4, 1e1)
            plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
            [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])

        plt.legend().remove() # This will remove the legend from the plot
        

        # Create a colorbar legend with custom tick labels
        norm = mpl.colors.Normalize(vmin=unique_deltas.min(), vmax=unique_deltas.max())
        sm = mpl.cm.ScalarMappable(cmap="viridis_r", norm=norm)
        sm.set_array([])  # Needed for the colorbar

        cbar = plt.colorbar(sm, ax=plt.gca())  # Attach colorbar
        cbar.set_label(r'$\Delta_{\mathrm{Fair}}$')  # Label for the colorbar

        # Set colorbar ticks at each distinct delta value
        cbar.set_ticks(unique_deltas)

        # Label only 0 and 0.5
        tick_labels = [r'$0$' if t == 0 else r'$0.5$' if t == 0.5 else '$0.25$' if t == 0.25 else "" for t in unique_deltas]
        cbar.set_ticklabels(tick_labels)

        # Increase the line width of the ticks
        cbar.ax.tick_params(width=2.5, length=6)  # Adjust width and length of ticks

        # Save the plot
        filename = f"./figures/{distr['budget_name']}_{distr['demand_name']}_efficiency_vs_M_{params['sigma_n']}_{params['sigma_b']}_{params['mu_n']}_{params['mu_b']}.pdf"
        plt.savefig(filename, bbox_inches='tight')
        # plt.show()
        plt.close('all')
