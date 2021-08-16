# PEL-Convergence
Comparison of the convergent properties of the PEL and Canonical Ensembles as they reach states of equilibrium. 

The PEL and Canonical Ensemble Main Codes folder, within the PEL and Canonical Ensembles folder, are the programs that run the actual ensembles and outputs data to .out files. These .out files can be named as desired, explaining which code is being saved as well as for how long (and other desired details such as move size and density). As an example, for the PEL ensemble using nonbiased trial moves sampled from a uniform distribution with a move size equivalent to a distance of 0.1 units and run for an equivalent temperature of T*=1 for 200,000 MC steps, the output file will look like: Nonbias_PEL_200000_Uniform_delta_0.1_T_1.out. By convention of the timeline when working on this project, the canonical ensemble was developed first as the standard Metropolis Monte Carlo sampling method. A number of models were tested in the making of this program, some with small numbers of particles and some starting off in random configurations, as opposed to the standard FCC lattice. A number of different densities were tested as well. As a result, these output files are named by the following: Metropolis_Large_MC_FCC_50000_RDF_5000_rho_1.2_delta_0.1_T_1.out, which just means that this is a canonical ensemble starting at an FCC lattice with 256 particles, run for 50,000 MC steps with 5,000 RDF calculation steps, a density of 1.2, a move step of 0.1 units and run at a temperature of T*=1. In the case of both the canonical and PEL ensembles, positions are printed out for RDF and energy calculculations every 10 MC steps. So, for a run of 50,000 MC steps, there will be 5,000 time slices for the RDF and energy calculations. Similarly, for the long runs of 200,000 MC steps, there will be 20,000 time slices for calculations. 

The rest of the folder for the PEL and Canonical Ensembles contains some raw positional data. This contains sphere placements so that the programs start from the same configurations. All of the programs use the Spheres_List_256.out file as their spheres configuration. The only two that utilize something different are the other two long PEL runs tested at T*=0.75, where three runs were tested to check similar convergence at the same low temperature example. Otherwise, everything else uses the first list of spheres mentioned above. These lists simply contain 0's and 1's representing sphere types. 0's represent type A spheres and 1's represent type B spheres. Additionally, this folder contains the output files from the main codes, tested at different temperatures and for different lengths of time. There are two main lengths of time used, 50,000 and 200,000 MC steps, where the long runs were used primarily to test the low temperature examples. There are four different temperatures tested: T*=0.725, 0.75, 1 and 2. As mentioned, in the PEL ensemble, the long run of T*=0.75 was tested three times for three different starting configurations, and there are therefore three different output files at this temperature for the long runs of 200,000 MC steps. The output files for each example are broken up into 8 different downloadable files, which can be combined afterwards to get the full output file. All results are calculated from these combined output files.

The rest of the data is stored within the energy and RDF folders. Beginning with the energy folder, labeled Energy Results, there are codes that calculate the energies and data that stores the main results used in the paper. The program called Raw_Energies.py calculates the energies from the raw data files in the PEL and Canonical Ensembles folder. The cumulative_energies.py program takes the cumulative average energy data stored in this folder, stored in the PEL Ensemble and Canonical Ensemble folders, and produces the comparison of a canonical and PEL ensemble example at the same temperature. As before, there are energy results for both ensembles run at temperatures of T*=0.725, 0.75, 1 and 2. This script produces the following figures in the manuscript: Fig. 2 in the main text, and Fig. S.1, S.4, and S.7 in the supplement. There is one figure in the manuscript, Fig. 5, which compares multiple energies at once. This figure is calculated from the script labeled Comparing_Multiple_Energies.py. Fig. 1 in the main text shows a comparison of reduced temperatures and the landscape energies for equivalent canonical and PEL ensemble configurations. This is produced from the script labeled Landscape_Energy_Plots.py. 
