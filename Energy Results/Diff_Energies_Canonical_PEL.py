import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

steps = 5000
n_skip = 10
# define the array of MC steps taken in the simulation
iterations = np.linspace(0,steps-1,steps)
iterations = np.array(iterations)

"""Define landscape energy from Kob-Andersen curve of best fit.
This is E_L = 2.639*T^0.6 - 8.656. This is the same value used
in both the canonical and PEL ensembles for comparisons of the
same temperature/energy values."""
T = 0.725
energy_land = -8.656 + 2.639*T**0.6
energy_land = np.round(energy_land, 3)

f = open('Cumulative_Energies_Canonical_T_0.725.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_met_0725 = []

for line in cumulative_energies:
    energies_cumul_ave_met_0725.append(float(line))

f.close()

# calculate difference between cumulative average energies and
# the landscape energy
energies_cumul_ave_met_diff_0725 = np.zeros(steps)
for i in range(steps):
    energies_cumul_ave_met_diff_0725[i] = energies_cumul_ave_met_0725[i] - energy_land

f = open('Cumulative_Energies_PEL_T_0.725.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_PEL_0725 = []

for line in cumulative_energies:
    energies_cumul_ave_PEL_0725.append(float(line))

f.close()

# Calculate the difference in cumulative average energies and
# the landscape energy
energies_cumul_ave_PEL_diff_0725 = np.zeros(steps)
for i in range(steps):
    energies_cumul_ave_PEL_diff_0725[i] = energies_cumul_ave_PEL_0725[i] - energy_land

"""Define landscape energy from Kob-Andersen curve of best fit.
This is E_L = 2.639*T^0.6 - 8.656. This is the same value used
in both the canonical and PEL ensembles for comparisons of the
same temperature/energy values."""
T = 1
energy_land = -8.656 + 2.639*T**0.6
energy_land = np.round(energy_land, 3)

f = open('Cumulative_Energies_Canonical_T_1.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_met_1 = []

for line in cumulative_energies:
    energies_cumul_ave_met_1.append(float(line))

f.close()

# calculate difference between cumulative average energies and
# the landscape energy
energies_cumul_ave_met_diff_1 = np.zeros(steps)
for i in range(steps):
    energies_cumul_ave_met_diff_1[i] = energies_cumul_ave_met_1[i] - energy_land

f = open('Cumulative_Energies_PEL_T_1.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_PEL_1 = []

for line in cumulative_energies:
    energies_cumul_ave_PEL_1.append(float(line))

f.close()

# Calculate the difference in cumulative average energies and
# the landscape energy
energies_cumul_ave_PEL_diff_1 = np.zeros(steps)
for i in range(steps):
    energies_cumul_ave_PEL_diff_1[i] = energies_cumul_ave_PEL_1[i] - energy_land

"""Define landscape energy from Kob-Andersen curve of best fit.
This is E_L = 2.639*T^0.6 - 8.656. This is the same value used
in both the canonical and PEL ensembles for comparisons of the
same temperature/energy values."""
T = 2
energy_land = -8.656 + 2.639*T**0.6
energy_land = np.round(energy_land, 3)

f = open('Cumulative_Energies_Canonical_T_2.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_met_2 = []

for line in cumulative_energies:
    energies_cumul_ave_met_2.append(float(line))

f.close()

# calculate difference between cumulative average energies and
# the landscape energy
energies_cumul_ave_met_diff_2 = np.zeros(steps)
for i in range(steps):
    energies_cumul_ave_met_diff_2[i] = energies_cumul_ave_met_2[i] - energy_land

f = open('Cumulative_Energies_PEL_T_2.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_PEL_2 = []

for line in cumulative_energies:
    energies_cumul_ave_PEL_2.append(float(line))

f.close()

# Calculate the difference in cumulative average energies and
# the landscape energy
energies_cumul_ave_PEL_diff_2 = np.zeros(steps)
for i in range(steps):
    energies_cumul_ave_PEL_diff_2[i] = energies_cumul_ave_PEL_2[i] - energy_land


# Color scheme using color-blind friendly colors
CB_color_cycle = ['c', 'm', 'sienna', 'k','y','salmon']

# Plot differences in cumulative average energies and E_L
fig,ax1 = plt.subplots()
ax1.plot(n_skip*iterations[:5000],energies_cumul_ave_met_diff_0725[:5000], 
    c = CB_color_cycle[0])
ax1.plot(n_skip*iterations[:5000],energies_cumul_ave_met_diff_1[:5000], 
    c = CB_color_cycle[1])
ax1.plot(n_skip*iterations[:5000],energies_cumul_ave_met_diff_2[:5000], 
    c = CB_color_cycle[2])
ax1.plot(n_skip*iterations,energies_cumul_ave_PEL_diff_0725, 
    c = CB_color_cycle[0], linestyle = 'dashed')
ax1.plot(n_skip*iterations,energies_cumul_ave_PEL_diff_1, c = CB_color_cycle[1], linestyle = 'dashed')
ax1.plot(n_skip*iterations,energies_cumul_ave_PEL_diff_2, c = CB_color_cycle[2], linestyle = 'dashed')
ax1.legend(('Canonical: T*=0.725','Canonical: T*=1','Canonical: T*=2',
    'PEL: T*=0.725','PEL: T*=1','PEL: T*=2'), 
    loc = 'lower right', prop = {"size":12})
plt.xlabel('MC Steps', fontsize = 15) # (n = {}, step size = {}$\sigma$)'.format(n_total,np.round(1/np.sqrt(2*alpha)/sigma, 2)), fontsize = 12.5)# t_relax))
plt.ylabel(r'Cumulative $(\langle V \rangle - E_{L})/(N\epsilon)$', 
    fontsize = 15)
plt.ylim(-1.0,1.0)

axins = zoomed_inset_axes(ax1, zoom = 2.5, bbox_to_anchor=(1050,950))
axins.plot(n_skip*iterations,energies_cumul_ave_met_diff_2, c = CB_color_cycle[2])
axins.plot(n_skip*iterations,energies_cumul_ave_PEL_diff_0725, 
    c = CB_color_cycle[0], linestyle = 'dashed')
axins.plot(n_skip*iterations,energies_cumul_ave_PEL_diff_1, c = CB_color_cycle[1], linestyle = 'dashed')
axins.plot(n_skip*iterations,energies_cumul_ave_PEL_diff_2, c = CB_color_cycle[2], linestyle = 'dashed')
axins.set_xlim(0,10000)
axins.set_ylim(-0.2,0.025)
mark_inset(ax1, axins, loc1 = 2, loc2 = 4)

plt.show()