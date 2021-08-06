# Import necessary modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""Read the file with the Metropolis Kob-Anderson LJ Binary 
Mixture Data using the Canonical Ensemble sampling method."""
f = open('Metropolis_Large_MC_FCC_200000_RDF_20000_rho_1.2_delta_0.1_T_0.725.out', 'r')

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

# relazation time if one is given before-hand (this is usually 0)
t_relax = contents[35]
t_relax = int(t_relax)

# length and volume of box
box_length = 10
V = box_length**3

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

# define exact densities of each sphere type
rho_a = spheres_a/V
rho_b = spheres_b/V

# define this value to simplify reading lines
skip = RDF_ideal

"""Define matrices which will store all x, y, and z coordinates
per time slice as row vectors. This runs through the total
number of time slices up to RDF_ideal."""
xs_new = np.zeros((RDF_ideal,n_total))
ys_new = np.zeros((RDF_ideal,n_total))
zs_new = np.zeros((RDF_ideal,n_total))

# read in all x, y, and z coordinates stored as row vectors in
# the matrices above
for i in range(0,RDF_ideal):
	xs_new[i] = contents[(4+3*i)*n_total+(45+3*i):(5+3*i)*n_total+(45+3*i)]
	ys_new[i] = contents[(5+3*i)*n_total+(46+3*i):(6+3*i)*n_total+(46+3*i)]
	zs_new[i] = contents[(6+3*i)*n_total+(47+3*i):(7+3*i)*n_total+(47+3*i)]

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
per_accept = contents[(7+3*skip)*n_total+(52+3*skip)]
per_accept = float(per_accept)
per_accept = np.round(per_accept, 1)

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

# define and read the radius values used in the RDF function
# and the final total averaged RDF 
radius = []
radii = contents[(7+3*skip)*n_total+(58+3*skip):(7+3*skip)*n_total+(58+3*skip)+Nbins]

for line in radii:
    radius.append(float(line))

f.close() # close file

# define all radii, including normalized radii w.r.t. 
# sphere-type distances
radius = np.array(radius)
radius_sigma_aa = radius/sigma_aa
radius_sigma_bb = radius/sigma_bb
radius_sigma_ab = radius/sigma_ab

"""Define the number of times preferable to skip through
time slices. Each time slice is every 10 MC steps when 
skip = 1. Increasing skip by one skips another 10 MC steps.""" 
skip = 1

# The total number of time slices divided by the skip value
# gives the total number of time slices that will be used
RDF_size = int(RDF_ideal/skip)

"""Define matrices to store RDFs. The matrix size is the number 
of bins in the histogram by the total RDF number used (stored
as the RDF_size above).""" 
g_aa_total = np.zeros((Nbins,RDF_size)) # array that stores the sum of distribution functions of A-type spheres
g_bb_total = np.zeros((Nbins,RDF_size)) # array that stores the sum of distribution functions of B-type spheres
g_ab_total = np.zeros((Nbins,RDF_size)) # array that stores the sum of distribution functions of A-B type spheres

"""Run through the desired number of time slices. For example,
a typical program runs through 50,000 MC steps. With every
10 MC steps skipped (skip parameter above is one), the 
desired number of time slices is 5,000."""
RDF_time = 5000

Ngr = 0 # Set total number of iterations initially to zero
delg = np.sqrt(3)*box_length/(2*Nbins) # width of each bin

"""Create RDFs for different combinations of sphere 
types (G_aa, G_ab, and G_bb) for every desired iteration,
taking into account the RDF_time and skip parameter."""
for i in range(0,RDF_time):
    if i%skip == 0:
        Ngr += 1 # count the RDF calculation number
        print(Ngr)
        vb = np.zeros(Nbins)
        g_aa = np.zeros(Nbins)
        g_bb = np.zeros(Nbins)
        g_ab = np.zeros(Nbins)
        nid_aa = np.zeros(Nbins)
        nid_bb = np.zeros(Nbins)
        nid_ab = np.zeros(Nbins)
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
                        # add counts to radial distribution function
                        ig_aa = int(r_aa_now/delg)
                        g_aa[ig_aa-1] += 2
                
                    # calculate RDF for type B-B spheres
                    elif spheres[j] == 1:
                        xr_bb_now = xs_new[i,j] - xs_new[i,k]  
                        yr_bb_now = ys_new[i,j] - ys_new[i,k]
                        zr_bb_now = zs_new[i,j] - zs_new[i,k]
                        xr_bb_now = xr_bb_now - box_length*np.round(xr_bb_now/box_length)
                        yr_bb_now = yr_bb_now - box_length*np.round(yr_bb_now/box_length)
                        zr_bb_now = zr_bb_now - box_length*np.round(zr_bb_now/box_length)
                        r_bb_now = np.sqrt(xr_bb_now**2 + yr_bb_now**2 + zr_bb_now**2)
                        # add counts to radial distribution function
                        ig_bb = int(r_bb_now/delg)
                        g_bb[ig_bb-1] += 2
            
                # calculate RDF for type A-B spheres
                elif spheres[j] != spheres[k]:
                    xr_ab_now = xs_new[i,j] - xs_new[i,k]  
                    yr_ab_now = ys_new[i,j] - ys_new[i,k]
                    zr_ab_now = zs_new[i,j] - zs_new[i,k]
                    xr_ab_now = xr_ab_now - box_length*np.round(xr_ab_now/box_length)
                    yr_ab_now = yr_ab_now - box_length*np.round(yr_ab_now/box_length)
                    zr_ab_now = zr_ab_now - box_length*np.round(zr_ab_now/box_length)
                    r_ab_now = np.sqrt(xr_ab_now**2 + yr_ab_now**2 + zr_ab_now**2)
                    # add counts to radial distribution function
                    ig_ab = int(r_ab_now/delg)
                    g_ab[ig_ab-1] += 2

        # Normalize radial distribution functions
        for h in range(0,Nbins):
            vb[h] = (4/3)*np.pi*((h+1)**3 - h**3)*delg**3
            nid_aa[h] = vb[h]*rho_a
            nid_bb[h] = vb[h]*rho_b
            nid_ab[h] = vb[h]*rho
            g_aa[h] = g_aa[h]/(nid_aa[h]*spheres_a)
            g_bb[h] = g_bb[h]/(nid_bb[h]*spheres_b)
            g_ab[h] = g_ab[h]/(nid_ab[h]*n_total*np.sqrt(x_a*x_b))
            # Add radial distribution functions      
            g_aa_total[h,int(i/skip)] = g_aa[h]
            g_bb_total[h,int(i/skip)] = g_bb[h]
            g_ab_total[h,int(i/skip)] = g_ab[h]

"""Calculate RDFs of type b-b spheres. This runs through all 
values in the bins for all desired time intervals. This takes
the skipping parameter into account."""

f = open('50000_Canonical_T_0.725_RDFs.out', 'w')

# G_bb for 10,000 MC steps
g_bb_10000_met = np.sum(g_bb_total[:,:int(1000/skip)], axis = 1)
g_bb_10000_met = g_bb_10000_met/int(1000/skip)

f.write('Gbb RDFs at 10,000 MC steps\n')
for val in g_bb_10000_met:
	f.write("{:}\n".format(val))

# G_bb for 20,000 MC steps
g_bb_20000_met = np.sum(g_bb_total[:,:int(2000/skip)], axis = 1)
g_bb_20000_met = g_bb_20000_met/int(2000/skip)

f.write('Gbb RDFs at 20,000 MC steps\n')
for val in g_bb_20000_met:
	f.write("{:}\n".format(val))

# G_bb for 30,000 MC steps
g_bb_30000_met = np.sum(g_bb_total[:,:int(3000/skip)], axis = 1)
g_bb_30000_met = g_bb_30000_met/int(3000/skip)

f.write('Gbb RDFs at 30,000 MC steps\n')
for val in g_bb_30000_met:
	f.write("{:}\n".format(val))

# G_bb for 40,000 MC steps
g_bb_40000_met = np.sum(g_bb_total[:,:int(4000/skip)], axis = 1)
g_bb_40000_met = g_bb_40000_met/int(4000/skip)

f.write('Gbb RDFs at 40,000 MC steps\n')
for val in g_bb_40000_met:
	f.write("{:}\n".format(val))

# G_bb for 50,000 MC steps
g_bb_50000_met = np.sum(g_bb_total[:,:int(5000/skip)], axis = 1)
g_bb_50000_met = g_bb_50000_met/int(5000/skip)

f.write('Gbb RDFs at 50,000 MC steps\n')
for val in g_bb_50000_met:
	f.write("{:}\n".format(val))

# # G_bb for 100,000 MC steps (applicable only for the long runs)
# g_bb_100000_met = np.sum(g_bb_total[:,:int(10000/skip)], axis = 1)
# g_bb_100000_met = g_bb_100000_met/int(10000/skip)

# f.write('Gbb RDFs at 100,000 MC steps\n')
# for val in g_bb_100000_met:
# 	f.write("{:}\n".format(val))

# # G_bb for 150,000 MC steps (applicable only for the long runs)
# g_bb_150000_met = np.sum(g_bb_total[:,:int(15000/skip)], axis = 1)
# g_bb_150000_met = g_bb_150000_met/int(15000/skip)

# f.write('Gbb RDFs at 150,000 MC steps\n')
# for val in g_bb_150000_met:
# 	f.write("{:}\n".format(val))

# # G_bb for 200,000 MC steps (applicable only for the long runs)
# g_bb_200000_met = np.sum(g_bb_total[:,:int(20000/skip)], axis = 1)
# g_bb_200000_met = g_bb_200000_met/int(20000/skip)

# f.write('Gbb RDFs at 200,000 MC steps\n')
# for val in g_bb_200000_met:
# 	f.write("{:}\n".format(val))

# Calculate RDFs of type a-a spheres.

# G_aa for 10,000 MC steps
g_aa_10000_met = np.sum(g_aa_total[:,:int(1000/skip)], axis = 1)
g_aa_10000_met = g_aa_10000_met/int(1000/skip)

f.write('Gaa RDFs at 10,000 MC steps\n')
for val in g_aa_10000_met:
	f.write("{:}\n".format(val))

# G_aa for 20,000 MC steps
g_aa_20000_met = np.sum(g_aa_total[:,:int(2000/skip)], axis = 1)
g_aa_20000_met = g_aa_20000_met/int(2000/skip)

f.write('Gaa RDFs at 20,000 MC steps\n')
for val in g_aa_20000_met:
	f.write("{:}\n".format(val))

# G_aa for 30,000 MC steps
g_aa_30000_met = np.sum(g_aa_total[:,:int(3000/skip)], axis = 1)
g_aa_30000_met = g_aa_30000_met/int(3000/skip)

f.write('Gaa RDFs at 30,000 MC steps\n')
for val in g_aa_30000_met:
	f.write("{:}\n".format(val))

# G_aa for 40,000 MC steps
g_aa_40000_met = np.sum(g_aa_total[:,:int(4000/skip)], axis = 1)
g_aa_40000_met = g_aa_40000_met/int(4000/skip)

f.write('Gaa RDFs at 40,000 MC steps\n')
for val in g_aa_40000_met:
	f.write("{:}\n".format(val))

# G_aa for 50,000 MC steps
g_aa_50000_met = np.sum(g_aa_total[:,:int(5000/skip)], axis = 1)
g_aa_50000_met = g_aa_50000_met/int(5000/skip)

f.write('Gaa RDFs at 50,000 MC steps\n')
for val in g_aa_50000_met:
	f.write("{:}\n".format(val))

# # G_aa for 100,000 MC steps (applicable only for the long runs)
# g_aa_100000_met = np.sum(g_aa_total[:,:int(10000/skip)], axis = 1)
# g_aa_100000_met = g_aa_100000_met/int(10000/skip)

# # G_aa for 150,000 MC steps (applicable only for the long runs)
# g_aa_150000_met = np.sum(g_aa_total[:,:int(15000/skip)], axis = 1)
# g_aa_150000_met = g_aa_150000_met/int(15000/skip)

# # G_aa for 200,000 MC steps (applicable only for the long runs)
# g_aa_200000_met = np.sum(g_aa_total[:,:int(20000/skip)], axis = 1)
# g_aa_200000_met = g_aa_200000_met/int(20000/skip)

# f.write('Gaa RDFs at 200,000 MC steps\n')
# for val in g_aa_200000_met:
# 	f.write("{:}\n".format(val))

# Calculate RDFs of type a-b spheres.

# G_ab for 10,000 MC steps
g_ab_10000_met = np.sum(g_ab_total[:,:int(1000/skip)], axis = 1)
g_ab_10000_met = g_ab_10000_met/int(1000/skip)

f.write('Gab RDFs at 10,000 MC steps\n')
for val in g_ab_10000_met:
	f.write("{:}\n".format(val))

# G_ab for 20,000 MC steps
g_ab_20000_met = np.sum(g_ab_total[:,:int(2000/skip)], axis = 1)
g_ab_20000_met = g_ab_20000_met/int(2000/skip)

f.write('Gab RDFs at 20,000 MC steps\n')
for val in g_ab_20000_met:
	f.write("{:}\n".format(val))

# G_ab for 30,000 MC steps
g_ab_30000_met = np.sum(g_ab_total[:,:int(3000/skip)], axis = 1)
g_ab_30000_met = g_ab_30000_met/int(3000/skip)

f.write('Gab RDFs at 30,000 MC steps\n')
for val in g_ab_30000_met:
	f.write("{:}\n".format(val))

# G_ab for 40,000 MC steps
g_ab_40000_met = np.sum(g_ab_total[:,:int(4000/skip)], axis = 1)
g_ab_40000_met = g_ab_40000_met/int(4000/skip)

f.write('Gab RDFs at 40,000 MC steps\n')
for val in g_ab_40000_met:
	f.write("{:}\n".format(val))

# G_ab for 50,000 MC steps
g_ab_50000_met = np.sum(g_ab_total[:,:int(5000/skip)], axis = 1)
g_ab_50000_met = g_ab_50000_met/int(5000/skip)

f.write('Gab RDFs at 50,000 MC steps\n')
for val in g_ab_50000_met:
	f.write("{:}\n".format(val))

# # G_ab for 100,000 MC steps (applicable only for the long runs)
# g_ab_100000_met = np.sum(g_ab_total[:,:int(10000/skip)], axis = 1)
# g_ab_100000_met = g_ab_100000_met/int(10000/skip)

# # G_ab for 150,000 MC steps (applicable only for the long runs)
# g_ab_150000_met = np.sum(g_ab_total[:,:int(15000/skip)], axis = 1)
# g_ab_150000_met = g_ab_150000_met/int(15000/skip)

# # G_ab for 200,000 MC steps (applicable only for the long runs)
# g_ab_200000_met = np.sum(g_ab_total[:,:int(20000/skip)], axis = 1)
# g_ab_200000_met = g_ab_200000_met/int(20000/skip)

# f.write('Gab RDFs at 200,000 MC steps\n')
# for val in g_ab_200000_met:
# 	f.write("{:}\n".format(val))

f.close()

"""Read the file with the Metropolis Kob-Anderson LJ Binary 
Mixture Data using the PEL sampling method."""
f = open('Nonbias_PEL_200000_MC_Uniform_delta_0.1_T_0.725_.out', 'r')

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

# length and volume of box
box_length = 10
V = box_length**3

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

"""Read in spheres list (this labels the particles with sphere
types a and b, where a value of 1 represents an a-type and a
value of 0 represents a b-type).""" 
spheres = []
sphere_list = contents[3*n_total+42:4*n_total+42]

for line in sphere_list:
	spheres.append(int(line))

# total a and b sphere numbers
A_spheres = contents[4*n_total+42]
spheres_a = contents[4*n_total+43]
spheres_a = float(spheres_a)
B_spheres = contents[4*n_total+44]
spheres_b = contents[4*n_total+45]
spheres_b = float(spheres_b)

# define exact densities of each sphere type
rho_a = spheres_a/V
rho_b = spheres_b/V

# define this value to simplify reading lines
skip = RDF_ideal

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

f.close()

"""Run through the desired number of time slices. For example,
a typical program runs through 50,000 MC steps. With every
10 MC steps skipped (skip parameter above is one), the 
desired number of time slices is 5,000."""
RDF_time = 5000

"""Define the number of times preferable to skip through
time slices. Each time slice is every 10 MC steps when 
skip = 1. Increasing skip by one skips another 10 MC steps.""" 
skip = 1

# The total number of time slices divided by the skip value
# gives the total number of time slices that will be used
RDF_size = int(RDF_ideal/skip)

"""Define matrices to store RDFs. The matrix size is the number 
of bins in the histogram by the total RDF number used (stored
as the RDF_size above).""" 
g_aa_total = np.zeros((Nbins,RDF_size)) # array that stores the sum of distribution functions of A-type spheres
g_bb_total = np.zeros((Nbins,RDF_size)) # array that stores the sum of distribution functions of B-type spheres
g_ab_total = np.zeros((Nbins,RDF_size)) # array that stores the sum of distribution functions of A-B type spheres

Ngr = 0 # Set total number of iterations initially to zero
delg = np.sqrt(3)*box_length/(2*Nbins) # width of each bin

"""Create RDFs for different combinations of sphere 
types (G_aa, G_ab, and G_bb) for every desired iteration,
taking into account the RDF_time and skip parameter."""
for i in range(0,RDF_time):
	if i%skip == 0:
	    Ngr += 1 # count the RDF calculation number
	    print(Ngr)
	    vb = np.zeros(Nbins)
	    g_aa = np.zeros(Nbins)
	    g_bb = np.zeros(Nbins)
	    g_ab = np.zeros(Nbins)
	    nid_aa = np.zeros(Nbins)
	    nid_bb = np.zeros(Nbins)
	    nid_ab = np.zeros(Nbins)
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
	                    # add counts to radial distribution function
	                    ig_aa = int(r_aa_now/delg)
	                    g_aa[ig_aa-1] += 2
	            
	                # calculate RDF for type B-B spheres
	                elif spheres[j] == 1:
	                    xr_bb_now = xs_new[i,j] - xs_new[i,k]  
	                    yr_bb_now = ys_new[i,j] - ys_new[i,k]
	                    zr_bb_now = zs_new[i,j] - zs_new[i,k]
	                    xr_bb_now = xr_bb_now - box_length*np.round(xr_bb_now/box_length)
	                    yr_bb_now = yr_bb_now - box_length*np.round(yr_bb_now/box_length)
	                    zr_bb_now = zr_bb_now - box_length*np.round(zr_bb_now/box_length)
	                    r_bb_now = np.sqrt(xr_bb_now**2 + yr_bb_now**2 + zr_bb_now**2)
	                    # add counts to radial distribution function
	                    ig_bb = int(r_bb_now/delg)
	                    g_bb[ig_bb-1] += 2
	        
	            # calculate RDF for type A-B spheres
	            elif spheres[j] != spheres[k]:
	                xr_ab_now = xs_new[i,j] - xs_new[i,k]  
	                yr_ab_now = ys_new[i,j] - ys_new[i,k]
	                zr_ab_now = zs_new[i,j] - zs_new[i,k]
	                xr_ab_now = xr_ab_now - box_length*np.round(xr_ab_now/box_length)
	                yr_ab_now = yr_ab_now - box_length*np.round(yr_ab_now/box_length)
	                zr_ab_now = zr_ab_now - box_length*np.round(zr_ab_now/box_length)
	                r_ab_now = np.sqrt(xr_ab_now**2 + yr_ab_now**2 + zr_ab_now**2)
	                # add counts to radial distribution function
	                ig_ab = int(r_ab_now/delg)
	                g_ab[ig_ab-1] += 2

	    # Normalize radial distribution functions
	    for h in range(0,Nbins):
	        vb[h] = (4/3)*np.pi*((h+1)**3 - h**3)*delg**3
	        nid_aa[h] = vb[h]*rho_a
	        nid_bb[h] = vb[h]*rho_b
	        nid_ab[h] = vb[h]*rho
	        g_aa[h] = g_aa[h]/(nid_aa[h]*spheres_a)
	        g_bb[h] = g_bb[h]/(nid_bb[h]*spheres_b)
	        g_ab[h] = g_ab[h]/(nid_ab[h]*n_total*np.sqrt(x_a*x_b))
	        # Add radial distribution functions      
	        g_aa_total[h,int(i/skip)] = g_aa[h]
	        g_bb_total[h,int(i/skip)] = g_bb[h]
	        g_ab_total[h,int(i/skip)] = g_ab[h]

f = open('50000_PEL_T_0.725_RDFs.out', 'w')

"""Calculate RDFs of all three sphere types. This runs through 
values in the bins for all desired time intervals. This takes
the skipping parameter into account."""

# G_bb for 10,000 MC steps
g_bb_10000_5 = np.sum(g_bb_total[:,:int(1000/skip)], axis = 1)
g_bb_10000_5 = g_bb_10000_5/int(1000/skip)

f.write('Gbb RDFs at 10,000 MC steps\n')
for val in g_bb_10000_5:
	f.write("{:}\n".format(val))

# G_bb for 20,000 MC steps
g_bb_20000_5 = np.sum(g_bb_total[:,:int(2000/skip)], axis = 1)
g_bb_20000_5 = g_bb_20000_5/int(2000/skip)

f.write('Gbb RDFs at 20,000 MC steps\n')
for val in g_bb_20000_5:
	f.write("{:}\n".format(val))

# G_bb for 30,000 MC steps
g_bb_30000_5 = np.sum(g_bb_total[:,:int(3000/skip)], axis = 1)
g_bb_30000_5 = g_bb_30000_5/int(3000/skip)

f.write('Gbb RDFs at 30,000 MC steps\n')
for val in g_bb_30000_5:
	f.write("{:}\n".format(val))

# G_bb for 40,000 MC steps
g_bb_40000_5 = np.sum(g_bb_total[:,:int(4000/skip)], axis = 1)
g_bb_40000_5 = g_bb_40000_5/int(4000/skip)

f.write('Gbb RDFs at 40,000 MC steps\n')
for val in g_bb_40000_5:
	f.write("{:}\n".format(val))

# G_bb for 50,000 MC steps
g_bb_50000_5 = np.sum(g_bb_total[:,:int(5000/skip)], axis = 1)
g_bb_50000_5 = g_bb_50000_5/int(5000/skip)

f.write('Gbb RDFs at 50,000 MC steps\n')
for val in g_bb_50000_5:
	f.write("{:}\n".format(val))

# # G_bb for 100,000 MC steps (applicable only for long-run times)
# g_bb_100000_5 = np.sum(g_bb_total[:,:int(10000/skip)], axis = 1)
# g_bb_100000_5 = g_bb_100000_5/int(10000/skip)

# f.write('Gbb RDFs at 100,000 MC steps\n')
# for val in g_bb_100000_5:
# 	f.write("{:}\n".format(val))

# # G_bb for 150,000 MC steps (applicable only for long-run times)
# g_bb_150000_5 = np.sum(g_bb_total[:,:int(15000/skip)], axis = 1)
# g_bb_150000_5 = g_bb_150000_5/int(15000/skip)

# f.write('Gbb RDFs at 150,000 MC steps\n')
# for val in g_bb_150000_5:
# 	f.write("{:}\n".format(val))

# # G_bb for 200,000 MC steps (applicable only for long-run times)
# g_bb_200000_5 = np.sum(g_bb_total[:,:int(20000/skip)], axis = 1)
# g_bb_200000_5 = g_bb_200000_5/int(20000/skip)

# f.write('Gbb RDFs at 200,000 MC steps\n')
# for val in g_bb_200000_5:
# 	f.write("{:}\n".format(val))

# Calculate RDFs for a-a spheres

# G_aa for 10,000 MC steps
g_aa_10000_5 = np.sum(g_aa_total[:,:int(1000/skip)], axis = 1)
g_aa_10000_5 = g_aa_10000_5/int(1000/skip)

f.write('Gaa RDFs at 10,000 MC steps\n')
for val in g_aa_10000_5:
	f.write("{:}\n".format(val))

# G_aa for 20,000 MC steps
g_aa_20000_5 = np.sum(g_aa_total[:,:int(2000/skip)], axis = 1)
g_aa_20000_5 = g_aa_20000_5/int(2000/skip)

f.write('Gaa RDFs at 20,000 MC steps\n')
for val in g_aa_20000_5:
	f.write("{:}\n".format(val))

# G_aa for 30,000 MC steps
g_aa_30000_5 = np.sum(g_aa_total[:,:int(3000/skip)], axis = 1)
g_aa_30000_5 = g_aa_30000_5/int(3000/skip)

f.write('Gaa RDFs at 30,000 MC steps\n')
for val in g_aa_30000_5:
	f.write("{:}\n".format(val))

# G_aa for 40,000 MC steps
g_aa_40000_5 = np.sum(g_aa_total[:,:int(4000/skip)], axis = 1)
g_aa_40000_5 = g_aa_40000_5/int(4000/skip)

f.write('Gaa RDFs at 40,000 MC steps\n')
for val in g_aa_40000_5:
	f.write("{:}\n".format(val))

# G_aa for 50,000 MC steps
g_aa_50000_5 = np.sum(g_aa_total[:,:int(5000/skip)], axis = 1)
g_aa_50000_5 = g_aa_50000_5/int(5000/skip)

f.write('Gaa RDFs at 50,000 MC steps\n')
for val in g_aa_50000_5:
	f.write("{:}\n".format(val))

# # G_aa for 100,000 MC steps (applicable only for long-run times)
# g_aa_100000_5 = np.sum(g_aa_total[:,:int(10000/skip)], axis = 1)
# g_aa_100000_5 = g_aa_100000_5/int(10000/skip)

# # G_aa for 150,000 MC steps (applicable only for long-run times)
# g_aa_150000_5 = np.sum(g_aa_total[:,:int(15000/skip)], axis = 1)
# g_aa_150000_5 = g_aa_150000_5/int(15000/skip)

# # G_aa for 200,000 MC steps (applicable only for long-run times)
# g_aa_200000_5 = np.sum(g_aa_total[:,:int(20000/skip)], axis = 1)
# g_aa_200000_5 = g_aa_200000_5/int(20000/skip)

# f.write('Gaa RDFs at 200,000 MC steps\n')
# for val in g_aa_200000_5:
# 	f.write("{:}\n".format(val))

# Calculate RDFs for a-b spheres

# G_ab for 10,000 MC steps
g_ab_10000_5 = np.sum(g_ab_total[:,:int(1000/skip)], axis = 1)
g_ab_10000_5 = g_ab_10000_5/int(1000/skip)

f.write('Gab RDFs at 10,000 MC steps\n')
for val in g_ab_10000_5:
	f.write("{:}\n".format(val))

# G_ab for 20,000 MC steps
g_ab_20000_5 = np.sum(g_ab_total[:,:int(2000/skip)], axis = 1)
g_ab_20000_5 = g_ab_20000_5/int(2000/skip)

f.write('Gab RDFs at 20,000 MC steps\n')
for val in g_ab_20000_5:
	f.write("{:}\n".format(val))

# G_ab for 30,000 MC steps
g_ab_30000_5 = np.sum(g_ab_total[:,:int(3000/skip)], axis = 1)
g_ab_30000_5 = g_ab_30000_5/int(3000/skip)

f.write('Gab RDFs at 30,000 MC steps\n')
for val in g_ab_30000_5:
	f.write("{:}\n".format(val))

# G_ab for 40,000 MC steps
g_ab_40000_5 = np.sum(g_ab_total[:,:int(4000/skip)], axis = 1)
g_ab_40000_5 = g_ab_40000_5/int(4000/skip)

f.write('Gab RDFs at 40,000 MC steps\n')
for val in g_ab_40000_5:
	f.write("{:}\n".format(val))

# G_ab for 50,000 MC steps
g_ab_50000_5 = np.sum(g_ab_total[:,:int(5000/skip)], axis = 1)
g_ab_50000_5 = g_ab_50000_5/int(5000/skip)

f.write('Gab RDFs at 50,000 MC steps\n')
for val in g_ab_50000_5:
	f.write("{:}\n".format(val))

# # G_ab for 100,000 MC steps (applicable only for long-run times)
# g_ab_100000_5 = np.sum(g_ab_total[:,:int(10000/skip)], axis = 1)
# g_ab_100000_5 = g_ab_100000_5/int(10000/skip)

# # G_ab for 150,000 MC steps (applicable only for long-run times)
# g_ab_150000_5 = np.sum(g_ab_total[:,:int(15000/skip)], axis = 1)
# g_ab_150000_5 = g_ab_150000_5/int(15000/skip)

# # G_ab for 200,000 MC steps (applicable only for long-run times)
# g_ab_200000_5 = np.sum(g_ab_total[:,:int(20000/skip)], axis = 1)
# g_ab_200000_5 = g_ab_200000_5/int(20000/skip)

# f.write('Gab RDFs at 200,000 MC steps\n')
# for val in g_ab_200000_5:
# 	f.write("{:}\n".format(val))

f.close()
