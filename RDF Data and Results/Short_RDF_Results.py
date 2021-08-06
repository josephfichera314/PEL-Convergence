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

"""Read in data from canonical ensemble example. This reads in
the RDF results for examples at T*=0.725, 0.75, 1 or 2."""

f = open('Short_Canonical_T_0.75_RDFs.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_10000 = contents[1:Nbins+1]

g_bb_20000 = contents[Nbins+2:2*Nbins+2]

g_bb_30000 = contents[2*Nbins+3:3*Nbins+3]

g_bb_40000 = contents[3*Nbins+4:4*Nbins+4]

g_bb_50000 = contents[4*Nbins+5:5*Nbins+5]

g_aa_50000 = contents[9*Nbins+10:10*Nbins+10]

g_ab_50000 = contents[14*Nbins+15:15*Nbins+15]


g_bb_10000_met = []
g_bb_20000_met = []
g_bb_30000_met = []
g_bb_40000_met = []
g_bb_50000_met = []
g_aa_50000_met = []
g_ab_50000_met = []

for line in g_bb_10000:
	g_bb_10000_met.append(float(line))

for line in g_bb_20000:
	g_bb_20000_met.append(float(line))

for line in g_bb_30000:
	g_bb_30000_met.append(float(line))

for line in g_bb_40000:
	g_bb_40000_met.append(float(line))

for line in g_bb_50000:
	g_bb_50000_met.append(float(line))

for line in g_aa_50000:
	g_aa_50000_met.append(float(line))

for line in g_ab_50000:
	g_ab_50000_met.append(float(line))

# Use the following two lists to obtain Fig. 3 in the manuscript.
# These are only needed for T*=0.75.
g_aa_20000 = contents[6*Nbins+7:7*Nbins+7]

g_ab_20000 = contents[11*Nbins+12:12*Nbins+12]

g_aa_20000_met = []
g_ab_20000_met = []

for line in g_aa_20000:
	g_aa_20000_met.append(float(line))

for line in g_ab_20000:
	g_ab_20000_met.append(float(line))

f.close()

"""Read in data from PEL ensemble example. This reads in
the RDF results for examples at T*=0.725, 0.75, 1 or 2."""

f = open('Short_PEL_T_0.75_RDFs.out', 'r')

# read in the contents
contents = f.readlines()

g_bb_10000 = contents[1:Nbins+1]

g_bb_20000 = contents[Nbins+2:2*Nbins+2]

g_bb_30000 = contents[2*Nbins+3:3*Nbins+3]

g_bb_40000 = contents[3*Nbins+4:4*Nbins+4]

g_bb_50000 = contents[4*Nbins+5:5*Nbins+5]

g_aa_50000 = contents[9*Nbins+10:10*Nbins+10]

g_ab_50000 = contents[14*Nbins+15:15*Nbins+15]

g_bb_10000_PEL = []
g_bb_20000_PEL = []
g_bb_30000_PEL = []
g_bb_40000_PEL = []
g_bb_50000_PEL = []
g_aa_50000_PEL = []
g_ab_50000_PEL = []

for line in g_bb_10000:
	g_bb_10000_PEL.append(float(line))

for line in g_bb_20000:
	g_bb_20000_PEL.append(float(line))

for line in g_bb_30000:
	g_bb_30000_PEL.append(float(line))

for line in g_bb_40000:
	g_bb_40000_PEL.append(float(line))

for line in g_bb_50000:
	g_bb_50000_PEL.append(float(line))

for line in g_aa_50000:
	g_aa_50000_PEL.append(float(line))

for line in g_ab_50000:
	g_ab_50000_PEL.append(float(line))

# Use the following two lists to obtain Fig. 3 in the manuscript.
# These are only needed for T*=0.75.
g_aa_20000 = contents[6*Nbins+7:7*Nbins+7]

g_ab_20000 = contents[11*Nbins+12:12*Nbins+12]

g_aa_20000_PEL = []
g_ab_20000_PEL = []

for line in g_aa_20000:
	g_aa_20000_PEL.append(float(line))

for line in g_ab_20000:
	g_ab_20000_PEL.append(float(line))

f.close()

radius = np.zeros(Nbins)
for i in range(0,Nbins):
    radius[i] = delg*(i+0.5)

radius_sigma_aa = radius/sigma_aa
radius_sigma_bb = radius/sigma_bb
radius_sigma_ab = radius/sigma_ab

"""3-panel plot showing the total RDF averages of a-a, a-b, and
b-b sphere types. This 3-panel plot compares the end-time RDFs 
of a single canonical ensemble and PEL ensemble example."""
plt.figure(figsize=(16,4))
plt.style.use('seaborn-colorblind')
plt.ylim(-0.1,2.0)
plt.tight_layout()
plt.ylabel('g(r)', fontsize = 17.5, rotation=0, labelpad = 20)

"""Replace 50,000 with 20,000 in the following figure to obtain
Fig. 3 in the manuscript. This will compare all three RDFs at 
20,000 MC steps. Keeping 50,000 MC steps obtains Fig. S.2,
S.5, and S.8 in the supplemental section."""

# G_aa plots
ax1 = plt.subplot(1,3,1)
ax1.plot(radius_sigma_aa[:180],g_aa_50000_met[:180])
ax1.plot(radius_sigma_aa[:180],g_aa_50000_PEL[:180])
ax1.legend(('Canonical','PEL'), loc = 'upper right', fontsize = 14)
ax1.set_xlabel('$r/\sigma_{aa}$', fontsize = 17.5)

# G_ab plots
ax2 = plt.subplot(1,3,2)
ax2.plot(radius_sigma_ab[:180],g_ab_50000_met[:180])
ax2.plot(radius_sigma_ab[:180],g_ab_50000_PEL[:180])
ax2.legend(('Canonical','PEL'), loc = 'upper right', fontsize = 14)
ax2.set_xlabel('$r/\sigma_{ab}$', fontsize = 17.5)

# G_bb plots
ax3 = plt.subplot(1,3,3)
ax3.plot(radius_sigma_bb[:180],g_bb_50000_met[:180])
ax3.plot(radius_sigma_bb[:180],g_bb_50000_PEL[:180])
ax3.legend(('Canonical','PEL'), loc = 'upper right', fontsize = 14)
ax3.set_xlabel('$r/\sigma_{bb}$', fontsize = 17.5)

"""This 2-panel plot compares how the G_bb RDFs of the two 
ensembles converge over time. Use this to obtain Fig. 4 and 6
in the main text, and Fig. S.3, S.6, and S.9 in the supplemental."""
plt.figure(figsize=(10,5))
plt.style.use('seaborn-colorblind')
plt.ylabel('$g_{bb}(r)$', fontsize = 17.5, rotation=0, labelpad = 25)

# Convergence of the canonical ensemble G_bb's
ax1 = plt.subplot(1,2,1)
ax1.plot(radius_sigma_bb[:180],g_bb_10000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_20000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_30000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_40000_met[:180])
ax1.plot(radius_sigma_bb[:180],g_bb_50000_met[:180])
ax1.legend(('10000 Can','20000 Can','30000 Can','40000 Can','50000 Can'), 
    title = "MC steps", loc = 'lower right', fontsize = 12.5)
ax1.set_xlabel('$r/\sigma_{bb}$', fontsize = 17.5)
plt.ylim(-0.1,1.45)

# Convergence of the PEL ensemble G_bb's
ax2 = plt.subplot(1,2,2)
ax2.plot(radius_sigma_bb[:180],g_bb_10000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_20000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_30000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_40000_PEL[:180])
ax2.plot(radius_sigma_bb[:180],g_bb_50000_PEL[:180])
ax2.legend(('10000 PEL','20000 PEL','30000 PEL','40000 PEL','50000 PEL'), 
    title = "MC steps", loc = 'lower right', fontsize = 12.5)
ax2.set_xlabel('$r/\sigma_{bb}$', fontsize = 17.5)
plt.ylim(-0.1,1.45)

plt.show()