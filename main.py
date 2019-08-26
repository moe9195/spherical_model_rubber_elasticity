import RubberElasticity as rs
import numpy as np
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore');

h, defType, distType, n, G, N, tolerance, std = 0.02, 1, 1, 10, 0.135, 10000, 0.9, 5

lam_max = np.linspace(2, 3, 3)
fig = plt.figure()
for i in lam_max:
    lam0, F, Fnew = rs.stress_strain_relation(n, N, G, h, std, tolerance, i, defType, distType)
    rs.plot_relation(lam0, F, Fnew, fig)
plt.show()
