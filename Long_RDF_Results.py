# Import necessary modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sigma = 1.673 # Baseline distance between spheres
sigma_aa = sigma # minimum distance between two A-type spheres
sigma_bb = 0.88*sigma # minimum distance between two B-type spheres
sigma_ab = 0.8*sigma # minimum distance between two A-B type spheres

Nbins = 300 # Number of bins
box_length = 10 # length of box
delg = np.sqrt(3)*box_length/(2*Nbins) # width of each bin

"""Open the T*=0.75 file of canonical ensemble to compare with
the PEL ensemble at the same temperature. Open the T*=0.725 
to compare with the PEL ensemble at the same temperature."""

f = open('Long_Canonical_T_0.725_RDFs.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_50000 = contents[1:Nbins+1]

g_bb_100000 = contents[Nbins+2:2*Nbins+2]

g_bb_150000 = contents[2*Nbins+3:3*Nbins+3]

g_bb_200000 = contents[3*Nbins+4:4*Nbins+4]

g_aa_200000 = contents[4*Nbins+5:5*Nbins+5]

g_ab_200000 = contents[5*Nbins+6:6*Nbins+6]

g_bb_50000_met = []
g_bb_100000_met = []
g_bb_150000_met = []
g_bb_200000_met = []
g_aa_200000_met = []
g_ab_200000_met = []

for line in g_bb_50000:
	g_bb_50000_met.append(float(line))

for line in g_bb_100000:
	g_bb_100000_met.append(float(line))

for line in g_bb_150000:
	g_bb_150000_met.append(float(line))

for line in g_bb_200000:
	g_bb_200000_met.append(float(line))

for line in g_aa_200000:
	g_aa_200000_met.append(float(line))

for line in g_ab_200000:
	g_ab_200000_met.append(float(line))

f.close()

"""Open the T*=0.75 file of PEL ensemble to compare with the 
canonical ensemble at the same temperature. Open the T*=0.725 
to compare with the canonical ensemble at the same temperature."""

f = open('Long_PEL_T_0.725_RDFs.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_50000 = contents[1:Nbins+1]

g_bb_100000 = contents[Nbins+2:2*Nbins+2]

g_bb_150000 = contents[2*Nbins+3:3*Nbins+3]

g_bb_200000 = contents[3*Nbins+4:4*Nbins+4]

g_aa_200000 = contents[4*Nbins+5:5*Nbins+5]

g_ab_200000 = contents[5*Nbins+6:6*Nbins+6]

g_bb_50000_PEL = []
g_bb_100000_PEL = []
g_bb_150000_PEL = []
g_bb_200000_PEL = []
g_aa_200000_PEL = []
g_ab_200000_PEL = []

for line in g_bb_50000:
	g_bb_50000_PEL.append(float(line))

for line in g_bb_100000:
	g_bb_100000_PEL.append(float(line))

for line in g_bb_150000:
	g_bb_150000_PEL.append(float(line))

for line in g_bb_200000:
	g_bb_200000_PEL.append(float(line))

for line in g_aa_200000:
	g_aa_200000_PEL.append(float(line))

for line in g_ab_200000:
	g_ab_200000_PEL.append(float(line))

f.close()

radius = np.zeros(Nbins)
for i in range(0,Nbins):
    radius[i] = delg*(i+0.5)

radius_sigma_aa = radius/sigma_aa
radius_sigma_bb = radius/sigma_bb
radius_sigma_ab = radius/sigma_ab

"""This 2-panel plot compares how the G_bb RDFs of the two 
ensembles converge over time. Use this plot to obtain 
Fig. S.10 and S.11 in the supplemental."""
plt.figure(figsize=(10,5))
plt.style.use('seaborn-colorblind')
plt.ylabel('$g_{bb}(r)$', fontsize = 17.5, rotation=0, labelpad = 25)

# Convergence of the canonical ensemble G_bb's
ax1 = plt.subplot(1,2,1)
ax1.plot(radius_sigma_bb[:180],g_bb_50000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_100000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_150000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_200000_met[:180])
ax1.legend(('50000 Can','100000 Can','150000 Can','200000 Can'),
    title = "MC steps", loc = 'lower right', fontsize = 12.5)
ax1.set_xlabel('$r/\sigma_{bb}$', fontsize = 17.5)
plt.ylim(-0.1,1.45)

# Convergence of the PEL ensemble G_bb's
ax2 = plt.subplot(1,2,2)
ax2.plot(radius_sigma_bb[:180],g_bb_50000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_100000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_150000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_200000_PEL[:180])
ax2.legend(('50000 PEL','100000 PEL','150000 PEL','200000 PEL'), 
    title = "MC steps", loc = 'lower right', fontsize = 12.5)
ax2.set_xlabel('$r/\sigma_{bb}$', fontsize = 17.5)
plt.ylim(-0.1,1.45)

"""Use the following three .out files to compare all three 
examples of the PEL ensemble at T*=0.75 with different starting 
configurations."""

f = open('Long_PEL_T_0.75_RDFs.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_200000 = contents[3*Nbins+4:4*Nbins+4]

g_aa_200000 = contents[4*Nbins+5:5*Nbins+5]

g_ab_200000 = contents[5*Nbins+6:6*Nbins+6]

g_bb_200000_PEL = []
g_aa_200000_PEL = []
g_ab_200000_PEL = []

for line in g_bb_200000:
	g_bb_200000_PEL.append(float(line))

for line in g_aa_200000:
	g_aa_200000_PEL.append(float(line))

for line in g_ab_200000:
	g_ab_200000_PEL.append(float(line))

f.close()

f = open('Long_PEL_T_0.75_RDFs_2.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_200000 = contents[3*Nbins+4:4*Nbins+4]

g_aa_200000 = contents[4*Nbins+5:5*Nbins+5]

g_ab_200000 = contents[5*Nbins+6:6*Nbins+6]

g_bb_200000_PEL_2 = []
g_aa_200000_PEL_2 = []
g_ab_200000_PEL_2 = []

for line in g_bb_200000:
	g_bb_200000_PEL_2.append(float(line))

for line in g_aa_200000:
	g_aa_200000_PEL_2.append(float(line))

for line in g_ab_200000:
	g_ab_200000_PEL_2.append(float(line))

f.close()

f = open('Long_PEL_T_0.75_RDFs_3.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_200000 = contents[3*Nbins+4:4*Nbins+4]

g_aa_200000 = contents[4*Nbins+5:5*Nbins+5]

g_ab_200000 = contents[5*Nbins+6:6*Nbins+6]

g_bb_200000_PEL_3 = []
g_aa_200000_PEL_3 = []
g_ab_200000_PEL_3 = []

for line in g_bb_200000:
	g_bb_200000_PEL_3.append(float(line))

for line in g_aa_200000:
	g_aa_200000_PEL_3.append(float(line))

for line in g_ab_200000:
	g_ab_200000_PEL_3.append(float(line))

f.close()

"""3-panel plot showing the total RDF averages of a-a, a-b, and
b-b sphere types. This 3-panel plot compares the end-time RDFs 
of all three examples of the PEL ensemble at T* = 0.75. Use
this plot to obtain Fig. S.12 in the supplemental."""
plt.figure(figsize=(16,4))
plt.style.use('seaborn-colorblind')
plt.ylim(-0.1,2.0)
plt.tight_layout()
plt.ylabel('g(r)', fontsize = 17.5, rotation=0, labelpad = 20)

# G_aa plots
ax1 = plt.subplot(1,3,1)
ax1.plot(radius_sigma_aa[:180],g_aa_200000_PEL[:180])
ax1.plot(radius_sigma_aa[:180],g_aa_200000_PEL_2[:180])
ax1.plot(radius_sigma_aa[:180],g_aa_200000_PEL_3[:180])
ax1.legend(('Configuration 1','Configuration 2','Configuration 3'), 
	loc = 'upper right', fontsize = 14)
ax1.set_xlabel('$r/\sigma_{aa}$', fontsize = 17.5)

# G_ab plots
ax2 = plt.subplot(1,3,2)
ax2.plot(radius_sigma_ab[:180],g_ab_200000_PEL[:180])
ax2.plot(radius_sigma_ab[:180],g_ab_200000_PEL_2[:180])
ax2.plot(radius_sigma_ab[:180],g_ab_200000_PEL_3[:180])
ax2.legend(('Configuration 1','Configuration 2','Configuration 3'), 
	loc = 'upper right', fontsize = 14)
ax2.set_xlabel('$r/\sigma_{ab}$', fontsize = 17.5)

# G_bb plots
ax3 = plt.subplot(1,3,3)
ax3.plot(radius_sigma_bb[:180],g_bb_200000_PEL[:180])
ax3.plot(radius_sigma_bb[:180],g_bb_200000_PEL_2[:180])
ax3.plot(radius_sigma_bb[:180],g_bb_200000_PEL_3[:180])
ax3.legend(('Configuration 1','Configuration 2','Configuration 3'), 
	loc = 'lower right', fontsize = 14)
ax3.set_xlabel('$r/\sigma_{bb}$', fontsize = 17.5)

plt.show()