import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

"""Read the file with the Metropolis Kob-Anderson LJ Binary 
Mixture Data using the Canonical Ensemble sampling method."""
f = open('Metropolis_Large_MC_FCC_50000_RDF_5000_rho_1.2_delta_0.1_T_1.out', 'r')

# read in the contents
contents = f.readlines()

# baseline sigma value
sigma = contents[1]
sigma = float(sigma)

# sigma_aa value (distance between a-a spheres)
sigma_aa = contents[3]
sigma_aa = float(sigma_aa)

# sigma_bb value (distance between b-b spheres)
sigma_bb = contents[5]
sigma_bb = float(sigma_bb)

# sigma_ab value (distance between a-b spheres)
sigma_ab = contents[7]
sigma_ab = float(sigma_ab)

# read in temperature of system
T = contents[9]
T = float(T)

# read in baseline energy constant epsilon
epsilon = contents[11]
epsilon = float(epsilon)

# epsilon_aa (energy constant between a-a spheres)
epsilon_aa = contents[13]
epsilon_aa = float(epsilon_aa)

# epsilon_bb (energy constant between b-b spheres)
epsilon_bb = contents[15]
epsilon_bb = float(epsilon_bb)

# epsilon_ab (energy constant between a-b spheres)
epsilon_ab = contents[17]
epsilon_ab = float(epsilon_ab)

# reduced density parameter of system, rho
rho = contents[19]
rho = float(rho)
rho = round(rho, 3)

# total number of spheres
n_total = contents[21]
n_total = int(n_total)

# fraction of a-type spheres (should be about 80%)
x_a = contents[23]
x_a = float(x_a)

# fraction of b-type spheres (should be about 20%)
x_b = contents[25]
x_b = float(x_b)
x_b = round(x_b, 2)

# packing fraction phi
phi = contents[27]
phi = float(phi)

# distance of the positive side of the uniform distribution
delta = contents[29]
delta = float(delta)

# total number of MC steps taken in the simulation
MC_steps = contents[31]
MC_steps = int(MC_steps)

"""Total number of RDF values averaged at the end w/out 
relaxation time (i.e. for 50,000 MC steps and skipping
averages for 10 MC steps, the number of RDFs averaged will
be 5,000 for the final averaged RDF.)"""
RDF_ideal = contents[33]
RDF_ideal = int(RDF_ideal)

# define this value to simplify reading lines
skip = RDF_ideal

# relazation time if one is given before-hand (this is usually 0)
t_relax = contents[35]
t_relax = int(t_relax)

"""Read in spheres list (this labels the particles with sphere
types a and b, where a value of 1 represents an a-type and a
value of 0 represents a b-type).""" 
spheres = []
sphere_list = contents[3*n_total+40:4*n_total+40]

for line in sphere_list:
	spheres.append(int(line))

# total a and b sphere numbers
A_spheres = contents[4*n_total+40]
spheres_a = contents[4*n_total+41]
spheres_a = float(spheres_a)
B_spheres = contents[4*n_total+42]
spheres_b = contents[4*n_total+43]
spheres_b = float(spheres_b)

# moves accepted
accept_statement = contents[(7+3*skip)*n_total+(47+3*skip)]
accept = contents[(7+3*skip)*n_total+(48+3*skip)]
accept = int(accept)

# moves rejected
reject_statement = contents[(7+3*skip)*n_total+(49+3*skip)]
reject = contents[(7+3*skip)*n_total+(50+3*skip)]
reject = int(reject)

# Acceptance percentage
per_accept_statement = contents[(7+3*skip)*n_total+(51+3*skip)]
per_accept_met = contents[(7+3*skip)*n_total+(52+3*skip)]
per_accept_met = float(per_accept_met)
per_accept_met = np.round(per_accept_met, 1)

# Exact RDF numbers (if the RDF_number = RDF_ideal, then the
# difference is zero)
RDF_statement = contents[(7+3*skip)*n_total+(53+3*skip)]
RDF_number = contents[(7+3*skip)*n_total+(54+3*skip)]
RDF_number = int(RDF_number)
RDF_difference = RDF_ideal - RDF_number

# number of bins used in the RDF histogram (usually 300)
Nbins_statement = contents[(7+3*skip)*n_total+(55+3*skip)]
Nbins = contents[(7+3*skip)*n_total+(56+3*skip)]
Nbins = int(Nbins)

# extract energies from canonical ensemble sampling simulation
energies_met = []
energy = contents[(7+3*skip)*n_total+4*Nbins+(62+3*skip):(7+3*skip)*n_total+4*Nbins+(62+3*skip)+RDF_number]

for line in energy:
	energies_met.append(float(line))

f.close()

# define array of MC steps taken in simulation
iterations_met = np.linspace(RDF_difference+0,RDF_ideal+0-1,RDF_number)
iterations_met = np.array(iterations_met)

"""Define landscape energy from Kob-Andersen curve of best fit.
This is E_L = 2.639*T^0.6 - 8.656. This is the same value used
in both the canonical and PEL ensembles for comparisons of the
same temperature/energy values."""
energy_land = -8.656 + 2.639*T**0.6
energy_land = np.round(energy_land, 3)
landscape_energy = np.zeros(RDF_number)
for i in range(0,RDF_number):
    landscape_energy[i] = energy_land

# define total average energies as an array
energies_met = np.array(energies_met)
energies_ave_met = energies_met/n_total

# calculate cumulative average energies
energies_cumul_met = np.zeros(RDF_number)
energies_cumul_ave_met = np.zeros(RDF_number)
for i in range(RDF_number):
	for j in range(i+1):
		energies_cumul_met[i] += energies_ave_met[j]

# final cumulative average energies
for i in range(RDF_number):
	energies_cumul_ave_met[i] = energies_cumul_met[i]/(i+1)

f = open('Cumulative_Energies_Canonical_T_1.out', 'w')

f.write('Cumulative Canonical Ensemble Energies at T*=1\n')
for val in energies_cumul_ave_met:
    f.write("{:}\n".format(val))

f.close()

# calculate the average energy of the average energies and 
# the standard deviation of the average energies
total_energy = 0
energy_diff = 0
for i in range(0,RDF_ideal):
    total_energy += energies_met[i]
mean_energy_met = total_energy/(n_total*RDF_ideal)
mean_energy_met = round(mean_energy_met, 3)
for i in range(0,RDF_ideal):
    energy_diff += (energies_ave_met[i] - mean_energy_met)**2
stand_dev = np.sqrt(energy_diff/((RDF_ideal-1)*RDF_ideal))
stand_dev = round(stand_dev, 3)
mean_energies_met = np.zeros(RDF_ideal)
for i in range(0,RDF_ideal):
    mean_energies_met[i] = mean_energy_met
    
# calculate average and standard deviation/error of cumulative
# average energies
energy_cumul_diff_met = 0
mean_cumul_energy_met = 0
stand_dev_met = 0
mean_cumul_energy_met = np.sum(energies_cumul_ave_met[0:RDF_ideal+1], 
    axis = 0)/(RDF_number)
for i in range(0,RDF_ideal):
    energy_cumul_diff_met += (energies_cumul_ave_met[i] - 
        mean_cumul_energy_met)**2
stand_dev_met = np.sqrt(energy_cumul_diff_met/(RDF_ideal-1))
stand_error_met = stand_dev_met/(np.sqrt(RDF_ideal))

"""Read the file with the Metropolis Kob-Anderson LJ Binary 
Mixture Data using the PEL sampling method."""
f = open('Nonbias_PEL_50000_MC_Uniform_delta_0.1_T_1_.out', 'r')

# read in the contents
contents = f.readlines()

# baseline sigma value
sigma = contents[1]
sigma = float(sigma)

# sigma_aa value (distance between a-a spheres)
sigma_aa = contents[3]
sigma_aa = float(sigma_aa)

# sigma_bb value (distance between b-b spheres)
sigma_bb = contents[5]
sigma_bb = float(sigma_bb)

# sigma_ab value (distance between a-b spheres)
sigma_ab = contents[7]
sigma_ab = float(sigma_ab)

# read in the energy-equivalent temperature of system
T = contents[9]
T = float(T)

# read in baseline energy constant epsilon
epsilon = contents[11]
epsilon = float(epsilon)

# epsilon_aa (energy constant between a-a spheres)
epsilon_aa = contents[13]
epsilon_aa = float(epsilon_aa)

# epsilon_bb (energy constant between b-b spheres)
epsilon_bb = contents[15]
epsilon_bb = float(epsilon_bb)

# epsilon_ab (energy constant between a-b spheres)
epsilon_ab = contents[17]
epsilon_ab = float(epsilon_ab)

# reduced density parameter of system, rho
rho = contents[19]
rho = float(rho)
rho = round(rho, 3)

# total number of spheres
n_total = contents[21]
n_total = int(n_total)

# fraction of a-type spheres (should be about 80%)
x_a = contents[23]
x_a = float(x_a)

# fraction of b-type spheres (should be about 20%)
x_b = contents[25]
x_b = float(x_b)
x_b = round(x_b, 2)

# packing fraction phi
phi = contents[27]
phi = float(phi)

"""Parameter used in the more complete version of the PEL model.
For the baseline PEL sampling scheme, this is always 0.5"""
a = contents[29]
a = float(a)

"""If the Gaussian distribution is used as the basis of the Monte
Carlo simulation, then this parameter is used. This parameter
is equal to 1/(2*variance). This parameter can be ignored when
the uniform distribution is employed."""
alpha = contents[31]
alpha = float(alpha)

# total number of MC steps taken in the simulation
MC_steps = contents[33]
MC_steps = int(MC_steps)

"""Total number of RDF values averaged at the end w/out 
relaxation time (i.e. for 50,000 MC steps and skipping
averages for 10 MC steps, the number of RDFs averaged will
be 5,000 for the final averaged RDF.)"""
RDF_ideal = contents[35]
RDF_ideal = int(RDF_ideal)

# relazation time if one is given before-hand (this is usually 0)
t_relax = contents[37]
t_relax = int(t_relax)

# length and volume of box
box_length = 10
V = box_length**3

# define this value to simplify reading lines
skip = RDF_ideal

"""Read in spheres list (this labels the particles with sphere
types a and b, where a value of 1 represents an a-type and a
value of 0 represents a b-type).""" 
spheres = []
sphere_list = contents[3*n_total+42:4*n_total+42]

for line in sphere_list:
	spheres.append(int(line))

"""Define matrices which will store all x, y, and z coordinates
per time slice as row vectors. This runs through the total
number of time slices up to RDF_ideal."""
xs_new = np.zeros((RDF_ideal,n_total))
ys_new = np.zeros((RDF_ideal,n_total))
zs_new = np.zeros((RDF_ideal,n_total))

# read in all x, y, and z coordinates stored as row vectors in
# the matrices above
for i in range(0,RDF_ideal):
	xs_new[i] = contents[(4+3*i)*n_total+(47+3*i):(5+3*i)*n_total+(47+3*i)]
	ys_new[i] = contents[(5+3*i)*n_total+(48+3*i):(6+3*i)*n_total+(48+3*i)]
	zs_new[i] = contents[(6+3*i)*n_total+(49+3*i):(7+3*i)*n_total+(49+3*i)]

# moves accepted
accept_statement = contents[(7+3*skip)*n_total+(49+3*skip)]
accept = contents[(7+3*skip)*n_total+(50+3*skip)]
accept = int(accept)

# moves rejected
reject_statement = contents[(7+3*skip)*n_total+(51+3*skip)]
reject = contents[(7+3*skip)*n_total+(52+3*skip)]
reject = int(reject)

# Acceptance percentage
per_accept_statement = contents[(7+3*skip)*n_total+(53+3*skip)]
per_accept = contents[(7+3*skip)*n_total+(54+3*skip)]
per_accept = float(per_accept)
per_accept = np.round(per_accept, 1)
per_accept_PEL = per_accept

# Exact RDF numbers (if the RDF_number = RDF_ideal, then the
# difference is zero)
RDF_statement = contents[(7+3*skip)*n_total+(55+3*skip)]
RDF_number = contents[(7+3*skip)*n_total+(56+3*skip)]
RDF_number = int(RDF_number)
RDF_difference = RDF_ideal - RDF_number

# number of bins used in the RDF histogram (usually 300)
Nbins_statement = contents[(7+3*skip)*n_total+(57+3*skip)]
Nbins = contents[(7+3*skip)*n_total+(58+3*skip)]
Nbins = int(Nbins)

f.close()

# define energies and set counting to 0
energies = []
Ngr = 0

# Calculate constants to be used in potentials and forces
epsilon_aa_4 = 4*epsilon_aa
epsilon_bb_4 = 4*epsilon_bb
epsilon_ab_4 = 4*epsilon_ab
epsilon_aa_24 = 24*epsilon_aa
epsilon_bb_24 = 24*epsilon_bb
epsilon_ab_24 = 24*epsilon_ab
sigma_aa_6 = sigma_aa**6
sigma_bb_6 = sigma_bb**6
sigma_ab_6 = sigma_ab**6
sigma_aa_12 = sigma_aa**12
sigma_bb_12 = sigma_bb**12
sigma_ab_12 = sigma_ab**12
sigma_aa_2_12 = 2*sigma_aa_12
sigma_bb_2_12 = 2*sigma_bb_12
sigma_ab_2_12 = 2*sigma_ab_12
V_aa_max = epsilon_aa_4*((2/5)**12 - (2/5)**6)
V_bb_max = epsilon_bb_4*((2/5)**12 - (2/5)**6)
V_ab_max = epsilon_ab_4*((2/5)**12 - (2/5)**6)

# Calculate energies based on all sphere-type interactions. 
# Append total energies to energies list.
for i in range(0,RDF_ideal):
    # if i%skip == 0: # relaxation time
    Ngr += 1 # count the RDF calculation number
    print(Ngr)
    V_LJ_total = 0
    for j in range(0,n_total-1):
        for k in range(j+1,n_total):
            if spheres[j] == spheres[k]:
                # calculate RDF for type A-A spheres
                if spheres[j] == 0:
                    xr_aa_now = xs_new[i,j] - xs_new[i,k]  
                    yr_aa_now = ys_new[i,j] - ys_new[i,k]
                    zr_aa_now = zs_new[i,j] - zs_new[i,k]
                    xr_aa_now = xr_aa_now - box_length*np.round(xr_aa_now/box_length)
                    yr_aa_now = yr_aa_now - box_length*np.round(yr_aa_now/box_length)
                    zr_aa_now = zr_aa_now - box_length*np.round(zr_aa_now/box_length)
                    r_aa_now = np.sqrt(xr_aa_now**2 + yr_aa_now**2 + zr_aa_now**2)
                    r_aa_now_6 = r_aa_now**6
                    r_aa_now_12 = r_aa_now**12
                    # Calculate energies
                    if r_aa_now > 2.5*sigma_aa:
                        V_LJ_now = 0
                    else:
                        V_LJ_now = epsilon_aa_4*((sigma_aa_12/r_aa_now_12) - 
                            (sigma_aa_6/r_aa_now_6)) - V_aa_max
                    V_LJ_total += V_LJ_now
            
                # calculate RDF for type B-B spheres
                elif spheres[j] == 1:
                    xr_bb_now = xs_new[i,j] - xs_new[i,k]  
                    yr_bb_now = ys_new[i,j] - ys_new[i,k]
                    zr_bb_now = zs_new[i,j] - zs_new[i,k]
                    xr_bb_now = xr_bb_now - box_length*np.round(xr_bb_now/box_length)
                    yr_bb_now = yr_bb_now - box_length*np.round(yr_bb_now/box_length)
                    zr_bb_now = zr_bb_now - box_length*np.round(zr_bb_now/box_length)
                    r_bb_now = np.sqrt(xr_bb_now**2 + yr_bb_now**2 + zr_bb_now**2)
                    r_bb_now_6 = r_bb_now**6
                    r_bb_now_12 = r_bb_now**12
                    # Calculate energies
                    if r_bb_now > 2.5*sigma_bb:
                        V_LJ_now = 0
                    else:
                        V_LJ_now = epsilon_bb_4*((sigma_bb_12/r_bb_now_12) - 
                            (sigma_bb_6/r_bb_now_6)) - V_bb_max
                    V_LJ_total += V_LJ_now
        
            # calculate RDF for type A-B spheres
            elif spheres[j] != spheres[k]:
                xr_ab_now = xs_new[i,j] - xs_new[i,k]  
                yr_ab_now = ys_new[i,j] - ys_new[i,k]
                zr_ab_now = zs_new[i,j] - zs_new[i,k]
                xr_ab_now = xr_ab_now - box_length*np.round(xr_ab_now/box_length)
                yr_ab_now = yr_ab_now - box_length*np.round(yr_ab_now/box_length)
                zr_ab_now = zr_ab_now - box_length*np.round(zr_ab_now/box_length)
                r_ab_now = np.sqrt(xr_ab_now**2 + yr_ab_now**2 + zr_ab_now**2)
                r_ab_now_6 = r_ab_now**6
                r_ab_now_12 = r_ab_now**12
                # Calculate energies
                if r_ab_now > 2.5*sigma_ab:
                    V_LJ_now = 0
                else:
                    V_LJ_now = epsilon_ab_4*((sigma_ab_12/r_ab_now_12) - 
                        (sigma_ab_6/r_ab_now_6)) - V_ab_max
                V_LJ_total += V_LJ_now

    # Append total energies to energies list
    energies.append(V_LJ_total)

# turn total energies into array
energies_PEL = np.array(energies)

"""Calculate average energies and set initial energy to same
initial energy as that in the canonical ensemble simulation.
Both simulations started at this energy."""
energies_ave_PEL = energies_PEL/n_total
energies_ave_PEL[0] = energies_ave_met[0]

# Calculate cumulative average energies from PEL ensemble
energies_cumul_ave_PEL = np.zeros(RDF_ideal)
for i in range(RDF_ideal):
	energies_cumul_ave_PEL[i] = np.sum(energies_ave_PEL[:i+1], 
        axis = 0)/(i+1)

f = open('Cumulative_Energies_PEL_T_1.out', 'w')

f.write('PEL Ensemble Cumulative Energies at T*=1\n')
for val in energies_cumul_ave_PEL:
    f.write("{:}\n".format(val))

f.close()

# define the array of MC steps taken in the simulation
iterations_PEL = np.linspace(RDF_difference,RDF_ideal-1,RDF_ideal)
iterations_PEL = np.array(iterations_PEL)

# calculate the average energy of the average energies and 
# the standard deviation of the average energies
total_energy = 0
energy_diff = 0
for i in range(0,RDF_ideal):
	total_energy += energies_PEL[i]
mean_energy_PEL = total_energy/(n_total*RDF_ideal)
mean_energy_PEL = round(mean_energy_PEL, 3)
for i in range(0,RDF_ideal):
	energy_diff += (energies_ave_PEL[i] - mean_energy_PEL)**2
stand_dev = np.sqrt(energy_diff/((RDF_ideal-1)*RDF_ideal))
stand_dev = round(stand_dev, 3)
mean_energies_PEL = np.zeros(RDF_ideal)
for i in range(0,RDF_ideal):
	mean_energies_PEL[i] = mean_energy_PEL
	
# calculate average and standard deviation/error of cumulative
# average energies
energy_cumul_diff_PEL = 0
mean_cumul_energy_PEL = 0
stand_dev_PEL = 0
mean_cumul_energy_PEL = np.sum(energies_cumul_ave_PEL[0:RDF_ideal+1], 
    axis = 0)/(RDF_number)
for i in range(0,RDF_ideal):
	energy_cumul_diff_PEL += (energies_cumul_ave_PEL[i] - 
        mean_cumul_energy_PEL)**2
stand_dev_PEL = np.sqrt(energy_cumul_diff_PEL/(RDF_ideal-1))
stand_error_PEL = stand_dev_PEL/(np.sqrt(RDF_ideal))

# Color scheme using color-blind friendly colors
CB_color_cycle = ['c', 'm', 'sienna', 'k','y','salmon']

# Ensures correct number of MC steps (this should be 10)
n_skip = MC_steps/RDF_ideal

# Plot average energies of canonical ensemble simulation
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
ax1 = plt.subplot(1,2,1)
plt.plot(n_skip*iterations_met[:],energies_ave_met[:],
 c = CB_color_cycle[0], alpha = 0.8)
plt.plot(n_skip*iterations_met[:],mean_energies_met[:], 
    c = CB_color_cycle[1])
plt.plot(n_skip*iterations_met[:],landscape_energy[:], 
    c = 'black')
plt.legend(('Canonical Ensemble <E>/N','Average Energy: {}'.
    format(mean_energy_met),'Landscape Energy: {}'.
    format(energy_land)), loc = 'lower center')
plt.xlabel('MC Steps')
plt.ylabel('Average Potential Energy per Particle')

# Plot average energies of PEL ensemble simulation
plt.subplot(1,2,2)
ax2 = plt.subplot(1,2,2)
plt.plot(n_skip*iterations_PEL[:],energies_ave_PEL[:], 
    c = CB_color_cycle[0], alpha = 0.8)
plt.plot(n_skip*iterations_PEL[:],mean_energies_PEL[:], 
    c = CB_color_cycle[1])
plt.plot(n_skip*iterations_PEL[:],landscape_energy[:RDF_number], 
    c = 'black')
plt.legend(('PEL Ensemble <E>/N','Average Energy: {}'.
    format(mean_energy_PEL),'Landscape Energy: {}'.
    format(energy_land)), loc = 'lower center')
plt.xlabel('MC Steps')

# Plot cumulative average energies
fig,ax1 = plt.subplots()
ax1.plot(n_skip*iterations_met[:5000],energies_cumul_ave_met[:5000], 
    c = CB_color_cycle[0])
ax1.plot(n_skip*iterations_PEL,energies_cumul_ave_PEL, 
    c = CB_color_cycle[1], linestyle = 'dashed')
ax1.plot(n_skip*iterations_met[:5000],landscape_energy[:5000], c = 'black')
ax1.legend(('Canonical','PEL','landscape energy: {}'.format(energy_land)),
    loc = 'lower right', prop = {"size":12})
plt.xlabel('MC Steps', fontsize = 15) # (n = {}, step size = {}$\sigma$)'.format(n_total,np.round(1/np.sqrt(2*alpha)/sigma, 2)), fontsize = 12.5)# t_relax))
plt.ylabel(r'Cumulative $(\langle V \rangle - E_{L})/(N\epsilon)$', 
    fontsize = 15)
plt.ylim(-6.5,-4.0) # use for T*=2
# plt.ylim(-6.5,-5.75) # use for T*=1
# plt.ylim(-6.5,-6.0) # use for T*=0.75
# plt.ylim(-6.55,-6.0) # use for T*=0.725

# adjust inset position as needed
axins = zoomed_inset_axes(ax1, zoom = 2.5, bbox_to_anchor=(1050,650))
axins.plot(n_skip*iterations_met[:5000],energies_cumul_ave_met[:5000], 
    c = CB_color_cycle[0]) # plot canonical within inset only for T*=2
axins.plot(n_skip*iterations_PEL,energies_cumul_ave_PEL, 
    c = CB_color_cycle[1], linestyle = 'dashed')
axins.plot(n_skip*iterations_met[:5000],landscape_energy[:5000], 
    c = 'black')
axins.set_xlim(0,10000)
axins.set_ylim(-5.0,-4.5) # use for T*=2
# axins.set_ylim(-6.05,-6.0) # use for T*=1
# axins.set_ylim(-6.46,-6.42) # use for T*=0.75 
# axins.set_ylim(-6.49,-6.47) # use for T*=0.725
mark_inset(ax1, axins, loc1 = 1, loc2 = 3)

plt.show()
