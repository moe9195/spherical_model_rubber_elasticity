import RubberElasticity as rs
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore');

'''
testing the model by plotting the stress-strain relation for an elastomer undergoing
uniaxial cyclic loading to 200%, 250%, and 300%.
'''

h, defType, distType, n, G, N, tolerance, std = 0.02, 1, 1, 10, 0.135, 10000, 0.9, 5
lam_max = np.linspace(2, 3, 3)
fig = plt.figure()
for x in lam_max:
    lam0, F, Fnew = rs.stress_strain_relation(n, N, G, h, std, tolerance, x, defType, distType)
    rs.plot_relation(lam0, F, Fnew, fig)
plt.show()
