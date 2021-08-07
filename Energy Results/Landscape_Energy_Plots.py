import numpy as np
import matplotlib.pyplot as plt

"""Plot of Landscape Energy vs Temperature. The landscape energy 
is equivalent to the relaxed average energies proposed by the 
Kob-Andersen papers. The equation is based on their line of best
fit for the energy-temperature relations."""
T = np.linspace(0.5,3,1000)
E = -8.656 + 2.639*T**0.6

"""Plot energy vs. temperature and highlight the main values we 
discuss in the paper (T* = 0.75, 1, and 2). This is Fig. 1 in the
main text."""
plt.figure()
plt.plot(T,E)
plt.plot(T[100], E[100], marker='o')
plt.vlines(T[100], -7, E[100], linestyle="dashed")
plt.hlines(E[100], 0, T[100], linestyle="dashed")
plt.text(0.0,-6.4,'-6.435')
plt.text(0.6,-7.2,'0.75')
plt.plot(T[200], E[200], marker='o')
plt.vlines(T[200], -7, E[200], linestyle="dashed")
plt.hlines(E[200], 0, T[200], linestyle="dashed")
plt.text(0.0,-5.95,'-6.017')
plt.text(0.925,-7.2,'1.0')
plt.plot(T[600], E[600], marker='o')
plt.vlines(T[600], -7, E[600], linestyle="dashed")
plt.hlines(E[600], 0, T[600], linestyle="dashed")
plt.text(0.0,-4.6,'-4.656')
plt.text(1.925,-7.2,'2.0')
plt.xlabel("T*", fontsize = 20)
plt.ylabel(r"Landscape Energy, $E_{L}/(N\epsilon)$", labelpad = 2, fontsize = 20)
plt.ylim(-7.3,-3.2)
plt.show()
