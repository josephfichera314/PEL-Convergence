import math
import numpy as np
import random as rnd

f = open('Metropolis_Large_MC_FCC_50000_RDF_5000_rho_1.2_delta_0.1_T_1.out', 'w')

# Define constants: minimum distance between particles (signma),
# density of particles (rho), and displacing distance (delta)
rho = 1.2 # reduced density (rho = n_total*sigma**3/V)
sigma = 1.673 # Baseline distance between spheres
sigma_aa = sigma # minimum distance between two A-type spheres
sigma_bb = 0.88*sigma # minimum distance between two B-type spheres
sigma_ab = 0.8*sigma # minimum distance between two A-B type spheres
T = 1 # reduced temperature (T = 1/beta*epsilon)
epsilon = 1 # Baseline energy constant in LJ Potential
epsilon_aa = epsilon # Energy constant for type A sphere interactions
epsilon_bb = 0.5*epsilon # Energy constant for type B sphere interactions
epsilon_ab = 1.5*epsilon # Energy constant for type A-B sphere interactions
beta = 1/(T*epsilon) # beta = 1/KT
x_a = 0.8 # fraction of type A spheres
x_b = 1.0 - x_a # fraction of type B spheres
phi = ((np.pi*rho)/6)*(x_a*sigma_aa**3 + x_b*sigma_bb**3) # packing fraction
delta = 0.1

# Calculate constants to be used in potentials and forces
epsilon_aa_4 = 4*epsilon_aa
epsilon_bb_4 = 4*epsilon_bb
epsilon_ab_4 = 4*epsilon_ab
sigma_aa_6 = sigma_aa**6
sigma_bb_6 = sigma_bb**6
sigma_ab_6 = sigma_ab**6
sigma_aa_12 = sigma_aa**12
sigma_bb_12 = sigma_bb**12
sigma_ab_12 = sigma_ab**12
V_aa_max = epsilon_aa_4*((2/5)**12 - (2/5)**6)
V_bb_max = epsilon_bb_4*((2/5)**12 - (2/5)**6)
V_ab_max = epsilon_ab_4*((2/5)**12 - (2/5)**6)

# Define dimensions of the box, number of particles, and iterations
axis_min = -5.0
axis_max = 5.0
box_length = axis_max - axis_min
V = box_length**3
n_total = int(rho*V/sigma**3) # total number of spheres obtained from density
MC_steps = 50000 # number of MC Steps
n_its = MC_steps*n_total # total number of iterations in terms of MC Steps
n_skip = 10*n_total # Number of iterations to skip for distribution function
RDF_ideal = int(n_its/n_skip)
t_relax = 0*n_total-1 # relaxation time

# Displace an initial set of n points on a grid in xyz space 
n_c = int(n_total/4) # number of spheres in each of the four cubes
num = int(np.cbrt(n_c)) # number of spheres on a single axis
delta_box = box_length/(2*num)
x_c = np.linspace(axis_min+delta_box,axis_max-delta_box,num)
y_c = np.linspace(axis_min+delta_box,axis_max-delta_box,num)
z_c = np.linspace(axis_min+delta_box,axis_max-delta_box,num)
xs_c, ys_c, zs_c = np.meshgrid(x_c, y_c, z_c)

# Reshape arrays of corner spheres to be used in moving particles
xs_c = np.reshape(xs_c, (n_c,1))
ys_c = np.reshape(ys_c, (n_c,1))
zs_c = np.reshape(zs_c, (n_c,1))

# Displace an initial set of spheres x-axis faces of cubes on a grid in xyz space
x_fx = np.linspace(axis_min+delta_box,axis_max-delta_box,num)
y_fx = np.linspace(axis_min+2*delta_box,axis_max,num)
z_fx = np.linspace(axis_min+2*delta_box,axis_max,num)
xs_fx, ys_fx, zs_fx = np.meshgrid(x_fx, y_fx, z_fx)

# Reshape arrays of spheres on x-axis faces to be used in moving particles
xs_fx = np.reshape(xs_fx, (n_c,1))
ys_fx = np.reshape(ys_fx, (n_c,1))
zs_fx = np.reshape(zs_fx, (n_c,1))

# Displace an initial set of spheres y-axis faces of cubes on a grid in xyz space
x_fy = np.linspace(axis_min+2*delta_box,axis_max,num)
y_fy = np.linspace(axis_min+delta_box,axis_max-delta_box,num)
z_fy = np.linspace(axis_min+2*delta_box,axis_max,num)
xs_fy, ys_fy, zs_fy = np.meshgrid(x_fy, y_fy, z_fy)

# Reshape arrays of spheres on y-axis faces to be used in moving particles
xs_fy = np.reshape(xs_fy, (n_c,1))
ys_fy = np.reshape(ys_fy, (n_c,1))
zs_fy = np.reshape(zs_fy, (n_c,1))

# Displace an initial set of spheres z-axis faces of cubes on a grid in xyz space
x_fz = np.linspace(axis_min+2*delta_box,axis_max,num)
y_fz = np.linspace(axis_min+2*delta_box,axis_max,num)
z_fz = np.linspace(axis_min+delta_box,axis_max-delta_box,num)
xs_fz, ys_fz, zs_fz = np.meshgrid(x_fz, y_fz, z_fz)

# Reshape arrays of spheres on z-axis faces to be used in moving particles
xs_fz = np.reshape(xs_fz, (n_c,1))
ys_fz = np.reshape(ys_fz, (n_c,1))
zs_fz = np.reshape(zs_fz, (n_c,1))

xs = np.concatenate((xs_c, xs_fx, xs_fy, xs_fz))
ys = np.concatenate((ys_c, ys_fx, ys_fy, ys_fz))
zs = np.concatenate((zs_c, zs_fx, zs_fy, zs_fz))

xs_init = np.zeros(n_total)
ys_init = np.zeros(n_total)
zs_init = np.zeros(n_total)

# Coordinates of all particles in FCC lattice
for i in range(0,n_total):
    xs_init[i] = xs[i]
    ys_init[i] = ys[i]
    zs_init[i] = zs[i]

f.write('Baseline Distance Betweem Spheres:\n')
f.write("{:}\n".format(sigma))

f.write('Minimum Distance Between Type A Spheres:\n')
f.write("{:}\n".format(sigma_aa))

f.write('Minimum Distance Between Type B Spheres:\n')
f.write("{:}\n".format(sigma_bb))

f.write('Minimum Distance Between Type A-B Sphere Pairs:\n')
f.write("{:}\n".format(sigma_ab))

f.write('Reduced Temperature of System:\n')
f.write("{:}\n".format(T))

f.write('Baseline Energy Constant:\n')
f.write("{:}\n".format(epsilon))

f.write('Energy Constant Betweem A-type Spheres:\n')
f.write("{:}\n".format(epsilon_aa))

f.write('Energy Constant Betweem B-type Spheres:\n')
f.write("{:}\n".format(epsilon_bb))

f.write('Energy Constant Betweem A-B type Spheres:\n')
f.write("{:}\n".format(epsilon_ab))

f.write('Total Density of System:\n')
f.write("{:}\n".format(rho))

f.write('Total Number of Spheres:\n')
f.write("{:}\n".format(n_total))

f.write('Fraction of Type A Spheres:\n')
f.write("{:}\n".format(x_a))

f.write('Fraction of Type B Spheres:\n')
f.write("{:}\n".format(x_b))

f.write('Total Packing Fraction:\n')
f.write("{:}\n".format(phi))

f.write('Step Size:\n')
f.write("{:}\n".format(delta))

f.write('MC Steps:\n')
f.write("{:}\n".format(MC_steps))

f.write('Ideal number of RDFs:\n')
f.write("{:}\n".format(RDF_ideal))

f.write('Relaxation Time (MC Steps):\n')
f.write("{:}\n".format(round(t_relax/n_total)))

f.write('Initial x-coordinates\n')
for valx in xs_init:
    f.write("{:}\n".format(valx))
f.write('Initial y-coordinates\n')
for valy in ys_init:
    f.write("{:}\n".format(valy))
f.write('Initial z-coordinates\n')
for valz in zs_init:
    f.write("{:}\n".format(valz))

# Keep track of sphere type; 0 is type A, 1 is type B
# Read the file with the Spheres Data
h = open('Spheres_List_256.out', 'r')

contents = h.readlines()

spheres = []
sphere_list = contents[1:n_total+1]

for line in sphere_list:
    spheres.append(int(line))

h.close()

spheres = np.array(spheres)

# Count number of each type of sphere
spheres_a = 0.0
spheres_b = 0.0
for i in range(n_total):
    if spheres[i] == 0:
        spheres_a += 1.0
    else:
        spheres_b += 1.0

# define densities of each sphere type
rho_a = spheres_a/V
rho_b = spheres_b/V

f.write('List of Spheres\n')
for s in spheres:
    f.write("{:}\n".format(s))

f.write('Exact Number of Type A Spheres:\n')
f.write("{:}\n".format(spheres_a))

f.write('Exact Number of Type B Spheres:\n')
f.write("{:}\n".format(spheres_b))

# Define initial reject counts as zero
reject = 0

# Define bin numbers in radial distribution functions and iterations
Nbins = 300 # Number of bins
Ngr = 0 # Set total number of iterations initially to zero
delg = np.sqrt(3)*box_length/(2*Nbins)
g_aa_total = np.zeros(Nbins) # array that stores the sum of distribution functions of type A spheres
g_bb_total = np.zeros(Nbins) # array that stores the sum of distribution functions of type B spheres
g_ab_total = np.zeros(Nbins) # array that stores the sum of distribution functions of A-B sphere pairs
radius = np.zeros(Nbins)
for i in range(0,Nbins):
    radius[i] = delg*(i+0.5)

# Set up list of energies
energies = []
# Calculate overall potential of initial system
V_LJ_init = 0
for j in range(0,n_total-1):
    for k in range(j+1,n_total):
        if spheres[j] == spheres[k]:
            # calculate RDF for type A-A spheres
            if spheres[j] == 0:
                xr_aa_init = xs_init[j] - xs_init[k]  
                yr_aa_init = ys_init[j] - ys_init[k]
                zr_aa_init = zs_init[j] - zs_init[k]
                xr_aa_init = xr_aa_init - box_length*np.round(xr_aa_init/box_length)
                yr_aa_init = yr_aa_init - box_length*np.round(yr_aa_init/box_length)
                zr_aa_init = zr_aa_init - box_length*np.round(zr_aa_init/box_length)
                r_aa_init = np.sqrt(xr_aa_init**2 + yr_aa_init**2 + zr_aa_init**2)
                r_aa_init_6 = r_aa_init**6
                r_aa_init_12 = r_aa_init**12
                # Calculate energies
                if r_aa_init > 2.5*sigma_aa:
                    V_init = 0
                else:
                    V_init = epsilon_aa_4*((sigma_aa_12/r_aa_init_12) - (sigma_aa_6/r_aa_init_6)) - V_aa_max
                V_LJ_init += V_init
                
            # calculate RDF for type B-B spheres
            elif spheres[j] == 1:
                xr_bb_init = xs_init[j] - xs_init[k]  
                yr_bb_init = ys_init[j] - ys_init[k]
                zr_bb_init = zs_init[j] - zs_init[k]
                xr_bb_init = xr_bb_init - box_length*np.round(xr_bb_init/box_length)
                yr_bb_init = yr_bb_init - box_length*np.round(yr_bb_init/box_length)
                zr_bb_init = zr_bb_init - box_length*np.round(zr_bb_init/box_length)
                r_bb_init = np.sqrt(xr_bb_init**2 + yr_bb_init**2 + zr_bb_init**2)
                r_bb_init_6 = r_bb_init**6
                r_bb_init_12 = r_bb_init**12
                # Calculate energies
                if r_bb_init > 2.5*sigma_bb:
                    V_init = 0
                else:
                    V_init = epsilon_bb_4*((sigma_bb_12/r_bb_init_12) - (sigma_bb_6/r_bb_init_6)) - V_bb_max
                V_LJ_init += V_init
            
        # calculate RDF for type A-B spheres
        elif spheres[j] != spheres[k]:
            xr_ab_init = xs_init[j] - xs_init[k]  
            yr_ab_init = ys_init[j] - ys_init[k]
            zr_ab_init = zs_init[j] - zs_init[k]
            xr_ab_init = xr_ab_init - box_length*np.round(xr_ab_init/box_length)
            yr_ab_init = yr_ab_init - box_length*np.round(yr_ab_init/box_length)
            zr_ab_init = zr_ab_init - box_length*np.round(zr_ab_init/box_length)
            r_ab_init = np.sqrt(xr_ab_init**2 + yr_ab_init**2 + zr_ab_init**2)
            r_ab_init_6 = r_ab_init**6
            r_ab_init_12 = r_ab_init**12
            # Calculate energies
            if r_ab_init > 2.5*sigma_ab:
                V_init = 0
            else:
                V_init = epsilon_ab_4*((sigma_ab_12/r_ab_init_12) - (sigma_ab_6/r_ab_init_6)) - V_ab_max
            V_LJ_init += V_init

print('initial energy: ', V_LJ_init)

# Iteratively displace point particles
for z in range(0,n_its):
    particle = int(math.floor(n_total*rnd.random())) # choose random particle to displace
    xs_new = np.zeros(n_total)
    ys_new = np.zeros(n_total)
    zs_new = np.zeros(n_total)
    for i in range(0,n_total):
        if i != particle:
            xs_new[i] = xs_init[i]
            ys_new[i] = ys_init[i]
            zs_new[i] = zs_init[i]
    xs_new[particle] = xs_init[particle] + delta*(rnd.random() - 0.5) # Displace x-coordinates
    ys_new[particle] = ys_init[particle] + delta*(rnd.random() - 0.5) # Displace y-coordinates
    zs_new[particle] = zs_init[particle] + delta*(rnd.random() - 0.5) # Displace z-coordinates

    # Periodic Boundary Conditions on chosen particle:
    # If particle exceeds a boundary, wrap around box
    xs_new = xs_new - box_length*np.round(xs_new/box_length)
    ys_new = ys_new - box_length*np.round(ys_new/box_length)
    zs_new = zs_new - box_length*np.round(zs_new/box_length)

    # Define acceptance/rejection scheme using 
    # LJ Potentials of Binary Sphere Mixture
    V_old, V_new = 0, 0
    for i in range(0,n_total):
        if i != particle:
            # Calculate all distances of initial configuration
            xr_old = xs_init[particle] - xs_init[i]
            yr_old = ys_init[particle] - ys_init[i]
            zr_old = zs_init[particle] - zs_init[i]
            xr_old = xr_old - box_length*np.round(xr_old/box_length)
            yr_old = yr_old - box_length*np.round(yr_old/box_length)
            zr_old = zr_old - box_length*np.round(zr_old/box_length)
            r_old = np.sqrt(xr_old**2 + yr_old**2 + zr_old**2)
            r_old_6 = r_old**6
            r_old_12 = r_old**12

            # Calculate all distances of new configuration
            xr_new = xs_new[particle] - xs_new[i]
            yr_new = ys_new[particle] - ys_new[i]
            zr_new = zs_new[particle] - zs_new[i]
            xr_new = xr_new - box_length*np.round(xr_new/box_length)
            yr_new = yr_new - box_length*np.round(yr_new/box_length)
            zr_new = zr_new - box_length*np.round(zr_new/box_length)
            r_new = np.sqrt(xr_new**2 + yr_new**2 + zr_new**2)
            r_new_6 = r_new**6
            r_new_12 = r_new**12

            if spheres[particle] == spheres[i]:
                # Calculate potentials of A-A sphere pairs
                if spheres[i] == 0:
                    # Old configuration calculations
                    if r_old > 2.5*sigma_aa:
                        V_aa_old = 0
                    else:
                        V_aa_old = epsilon_aa_4*((sigma_aa_12/r_old_12) - (sigma_aa_6/r_old_6)) - V_aa_max 
                    V_old += V_aa_old
                    # New configuration calculations
                    if r_new > 2.5*sigma_aa:
                        V_aa_new = 0
                    else:
                        V_aa_new = epsilon_aa_4*((sigma_aa_12/r_new_12) - (sigma_aa_6/r_new_6)) - V_aa_max
                    V_new += V_aa_new
                # Calculate potentials of B-B sphere pairs
                elif spheres[i] == 1:
                    # Old configuration calculations
                    if r_old > 2.5*sigma_bb:
                        V_bb_old = 0
                    else:
                        V_bb_old = epsilon_bb_4*((sigma_bb_12/r_old_12) - (sigma_bb_6/r_old_6)) - V_bb_max 
                    V_old += V_bb_old
                    # New configuration calculations
                    if r_new > 2.5*sigma_bb:
                        V_bb_new = 0
                    else:
                        V_bb_new = epsilon_bb_4*((sigma_bb_12/r_new_12) - (sigma_bb_6/r_new_6)) - V_bb_max
                    V_new += V_bb_new
            # Calculate potentials of A-B sphere pairs
            elif spheres[particle] != spheres[i]:
                # Old configuration calculations
                if r_old > 2.5*sigma_ab:
                    V_ab_old = 0
                else:
                    V_ab_old = epsilon_ab_4*((sigma_ab_12/r_old_12) - (sigma_ab_6/r_old_6)) - V_ab_max
                V_old += V_ab_old
                # New configuration calculations
                if r_new > 2.5*sigma_ab:
                    V_ab_new = 0
                else:
                    V_ab_new = epsilon_ab_4*((sigma_ab_12/r_new_12) - (sigma_ab_6/r_new_6)) - V_ab_max
                V_new += V_ab_new

   # Accept/Reject based on LJ Potential between small and big spheres
    delta_V = V_new - V_old
    V_LJ_new = V_LJ_init + delta_V
    if delta_V < 0:
        xs_new[particle] = xs_new[particle]
        ys_new[particle] = ys_new[particle]
        zs_new[particle] = zs_new[particle]
    elif rnd.random() < np.exp(-beta*delta_V):
        xs_new[particle] = xs_new[particle]
        ys_new[particle] = ys_new[particle]
        zs_new[particle] = zs_new[particle]
    else:
        reject += 1 
        xs_new[particle] = xs_init[particle]
        ys_new[particle] = ys_init[particle]
        zs_new[particle] = zs_init[particle]

    # Create RDF for different combinations of sphere 
    # sizes for every 'n_skip' iterations
    if z%n_skip == 0:
        f.write('New x-coordinates\n')
        for valx in xs_new:
            f.write("{:.6f}\n".format(valx))
        f.write('New y-coordinates\n')
        for valy in ys_new:
            f.write("{:.6f}\n".format(valy))
        f.write('New z-coordinates\n')
        for valz in zs_new:
            f.write("{:.6f}\n".format(valz))
        if z > t_relax: # relaxation time
            Ngr += 1 # count the RDF calculation number
            print(Ngr)
            print("initial energy: ", V_LJ_init)
            print("new energy: ", V_LJ_new)
            V_LJ_total = 0
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
                        # calculate RDF for small spheres
                        if spheres[j] == 0:
                            xr_aa_now = xs_new[j] - xs_new[k]  
                            yr_aa_now = ys_new[j] - ys_new[k]
                            zr_aa_now = zs_new[j] - zs_new[k]
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
                                V_LJ_now = epsilon_aa_4*((sigma_aa_12/r_aa_now_12) - (sigma_aa_6/r_aa_now_6)) - V_aa_max
                            V_LJ_total += V_LJ_now
                            # add counts to radial distribution function
                            ig_aa = int(r_aa_now/delg)
                            g_aa[ig_aa-1] += 2
                    
                        # calculate RDF for big spheres
                        elif spheres[j] == 1:
                            xr_bb_now = xs_new[j] - xs_new[k]  
                            yr_bb_now = ys_new[j] - ys_new[k]
                            zr_bb_now = zs_new[j] - zs_new[k]
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
                                V_LJ_now = epsilon_bb_4*((sigma_bb_12/r_bb_now_12) - (sigma_bb_6/r_bb_now_6)) - V_bb_max
                            V_LJ_total += V_LJ_now
                            # add counts to radial distribution function
                            ig_bb = int(r_bb_now/delg)
                            g_bb[ig_bb-1] += 2
                
                    # calculate RDF between big and small spheres
                    elif spheres[j] != spheres[k]:
                        xr_ab_now = xs_new[j] - xs_new[k]  
                        yr_ab_now = ys_new[j] - ys_new[k]
                        zr_ab_now = zs_new[j] - zs_new[k]
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
                            V_LJ_now = epsilon_ab_4*((sigma_ab_12/r_ab_now_12) - (sigma_ab_6/r_ab_now_6)) - V_ab_max
                        V_LJ_total += V_LJ_now
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
            g_aa_total += g_aa
            g_bb_total += g_bb
            g_ab_total += g_ab

            # Calculate energies
            energies.append(V_LJ_total) 

    # Reset this new list of coordinates as the next initial set of 
    # coordinates in order to iterate through this process of 
    # displacing particles.
    xs_init = xs_new
    ys_init = ys_new
    zs_init = zs_new
    V_LJ_init = V_LJ_new

# Average over the iterated distribution functions
g_aa_ave = np.zeros(Nbins)
g_bb_ave = np.zeros(Nbins)
g_ab_ave = np.zeros(Nbins)
for x in range(0,Nbins):
    g_aa_ave[x] = g_aa_total[x]/Ngr 
    g_bb_ave[x] = g_bb_total[x]/Ngr
    g_ab_ave[x] = g_ab_total[x]/Ngr

f.write('Final x-coordinates\n')
for valx in xs_new:
    f.write("{:.6f}\n".format(valx))
f.write('Final y-coordinates\n')
for valy in ys_new:
    f.write("{:.6f}\n".format(valy))
f.write('Final z-coordinates\n')
for valz in zs_new:
    f.write("{:.6f}\n".format(valz))

f.write('Number of accepted particle moves:\n')
f.write("{:}\n".format(n_its - reject))

f.write('Number of rejected particle moves:\n')
f.write("{:}\n".format(reject))

f.write('Percentage of accepted particle moves:\n')
f.write("{:}\n".format((n_its - reject)*100/n_its))

f.write('Total number of radial distribution functions:\n')
f.write("{:}\n".format(Ngr))

f.write('Number of bins in histrogram:\n')
f.write("{:}\n".format(Nbins))

f.write('Radius values:\n')
for val in radius:
    f.write("{:}\n".format(val))

f.write('Average Distribution Function of Type A Sphere Pairs:\n')
for val_aa in g_aa_ave:
    f.write("{:}\n".format(val_aa))

f.write('Average Distribution Function of Type B Sphere Pairs:\n')
for val_bb in g_bb_ave:
    f.write("{:}\n".format(val_bb))

f.write('Average Distribution Function of Type A-B Sphere Pairs:\n')
for val_ab in g_ab_ave:
    f.write("{:}\n".format(val_ab))

energies[0] = V_LJ_init

f.write('Total energies of system: \n')
for val in energies:
    f.write("{:}\n".format(val))

f.close()
