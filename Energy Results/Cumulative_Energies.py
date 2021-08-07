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
T = 2
energy_land = -8.656 + 2.639*T**0.6
energy_land = np.round(energy_land, 3)
landscape_energy = np.zeros(steps)
for i in range(0,steps):
    landscape_energy[i] = energy_land

f = open('Cumulative_Energies_Canonical_T_2.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_met = []

for line in cumulative_energies:
	energies_cumul_ave_met.append(float(line))

f.close()

f = open('Cumulative_Energies_PEL_T_2.out', 'r')

# read in the contents
contents = f.readlines()

cumulative_energies = contents[1:steps+1]

energies_cumul_ave_PEL = []

for line in cumulative_energies:
	energies_cumul_ave_PEL.append(float(line))

f.close()

# Color scheme using color-blind friendly colors
CB_color_cycle = ['c', 'm', 'sienna', 'k','y','salmon']

# Plot cumulative average energies
fig,ax1 = plt.subplots()
ax1.plot(n_skip*iterations,energies_cumul_ave_met, 
    c = CB_color_cycle[0])
ax1.plot(n_skip*iterations,energies_cumul_ave_PEL, 
    c = CB_color_cycle[1], linestyle = 'dashed')
ax1.plot(n_skip*iterations,landscape_energy, c = 'black')
ax1.legend(('Canonical','PEL','landscape energy: {}'.format(energy_land)),
    loc = 'upper right', prop = {"size":12})
plt.xlabel('MC Steps', fontsize = 15) # (n = {}, step size = {}$\sigma$)'.format(n_total,np.round(1/np.sqrt(2*alpha)/sigma, 2)), fontsize = 12.5)# t_relax))
plt.ylabel(r'Cumulative Average Potential Energy $\langle V \rangle/(N\epsilon)$', 
    fontsize = 15)
plt.ylim(-6.5,-4.0) # use for T*=2
# plt.ylim(-6.5,-5.75) # use for T*=1
# plt.ylim(-6.5,-6.0) # use for T*=0.75
# plt.ylim(-6.55,-6.0) # use for T*=0.725

# adjust inset position as needed
axins = zoomed_inset_axes(ax1, zoom = 2.5, bbox_to_anchor=(1050,650))
axins.plot(n_skip*iterations,energies_cumul_ave_met, 
    c = CB_color_cycle[0]) # plot canonical within inset only for T*=2
axins.plot(n_skip*iterations,energies_cumul_ave_PEL, 
    c = CB_color_cycle[1], linestyle = 'dashed')
axins.plot(n_skip*iterations,landscape_energy, 
    c = 'black')
axins.set_xlim(0,10000)
axins.set_ylim(-5.0,-4.5) # use for T*=2
# axins.set_ylim(-6.05,-6.0) # use for T*=1
# axins.set_ylim(-6.46,-6.42) # use for T*=0.75 
# axins.set_ylim(-6.49,-6.47) # use for T*=0.725
mark_inset(ax1, axins, loc1 = 1, loc2 = 3)

plt.show()