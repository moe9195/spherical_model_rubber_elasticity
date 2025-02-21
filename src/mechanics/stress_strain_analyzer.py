from math import sqrt, log
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from ..enums.deformation_type import DeformationType
from ..enums.distribution_type import DistributionType
from ..models.network_parameters import NetworkParameters
from ..geometry.sphere_triangulator import SphereTriangulator
from .chain_network import ChainNetwork
from .deformation_calculator import DeformationCalculator
from .force_calculator import ForceCalculator

class StressStrainAnalyzer:
    def __init__(self, params: NetworkParameters):
        self.params = params

    def calculate_relation(self, lam_max: float, def_type: DeformationType, 
                         dist_type: DistributionType) -> Tuple[NDArray, NDArray, NDArray]:
        np.random.seed(1)
        lam = np.arange(1, lam_max + self.params.h, self.params.h)

        if dist_type == DistributionType.UNIFORM_GRID:
            N0 = round(1 + (sqrt(self.params.N) / sqrt(2)))
            M = SphereTriangulator.sphere_triangulation_grid(N0, N0)
        else:
            N0 = round(log(self.params.N / 8) / (2 * log(2)))
            M = SphereTriangulator.sphere_triangulation_octahedron(N0, 1)

        omega0 = ChainNetwork.solid_angle(M)
        Points = ChainNetwork.spherical_triangle_centroid(M)
        rd = DeformationCalculator.deform(lam, def_type)
        
        rx = np.outer(rd[0, :], Points[0, :])
        ry = np.outer(rd[1, :], Points[1, :])
        rz = np.outer(rd[2, :], Points[2, :])
        r = np.sqrt((rx ** 2) + (ry ** 2) + (rz ** 2))
        
        wd, ln = r.shape[0], r.shape[1]

        mu = log(self.params.n / sqrt(1 + ((self.params.std ** 2) / (self.params.n ** 2))))
        sig = sqrt(log(1 + ((self.params.std ** 2) / (self.params.n ** 2))))
        n0 = np.random.lognormal(mean=mu, sigma=sig, size=ln)

        r[r > np.sqrt(n0) * self.params.tolerance] = 0
        s, f, F = ForceCalculator.force(r, omega0, n0, self.params.G, self.params.h)

        a0 = r[wd - 1, :]
        rnew = r[:, ~(a0 == 0)]
        omeganew = omega0[0, ~(a0 == 0)]
        nnew = n0[~(a0 == 0)]
        _, _, Fnew = ForceCalculator.force(rnew, omeganew, nnew, self.params.G, self.params.h)
        
        lam0 = 100 * np.delete(lam, -1)
        return lam0, F, Fnew