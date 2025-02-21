import RubberElasticity as rs
import numpy as np
import matplotlib.pyplot as plt
from RubberElasticity import NetworkParameters, DeformationType, DistributionType, StressStrainAnalyzer

np.seterr(divide='ignore', invalid='ignore')

'''
testing the model by plotting the stress-strain relation for an elastomer undergoing
uniaxial cyclic loading to 200%, 250%, and 300%.
'''

# Initialize network parameters
params = NetworkParameters(
    n=10,           # average number of links per chain
    N=10000,        # chain density per unit sphere
    G=0.135,        # shear modulus
    h=0.02,         # step size
    tolerance=0.9,   # maximum percent tolerance
    std=5           # standard deviation
)

# Create analyzer
analyzer = StressStrainAnalyzer(params)

# Test different maximum stretch ratios
lam_max = np.linspace(2, 3, 3)
fig = plt.figure()

for x in lam_max:
    lam0, F, Fnew = analyzer.calculate_relation(
        lam_max=x,
        def_type=DeformationType.UNIAXIAL,
        dist_type=DistributionType.UNIFORM_GRID
    )
    rs.PlotManager.plot_relation(lam0, F, Fnew, fig)

plt.show()
