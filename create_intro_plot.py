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

# Use seaborn style
sns.set_style("white")
sns.set_palette(sns.color_palette("viridis")[::-1])

# Range of delta values
delta = np.linspace(0, 3, 100)

# M values to plot
M_values = [3, 5, 7]

# Color palette
palette = sns.color_palette("viridis", n_colors=len(M_values))[::-1]

# Create figure
plt.figure(figsize=(8, 6))

# Plot the curves for different M
for idx, M in enumerate(M_values):
    inefficiency = (1 / M) * np.exp(-delta * M)
    plt.plot(delta, inefficiency, color=palette[idx], label=f'M = {M}')

# Add labels, title, and legend
plt.xlabel(r'$\Delta_{\mathrm{Fair}}$')
plt.ylabel(r'$\Delta_{\mathrm{Efficiency}}$')

plt.xticks([])
plt.yticks([])


plt.legend().remove() # This will remove the legend from the plot

# plt.title('Theoretical Tradeoff: Inefficiency = (1/M) * exp(-Δ * M)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

filename = f"./figures/intro_tradeoff.pdf"
plt.savefig(filename, bbox_inches='tight')
#     # plt.show()
plt.close('all')