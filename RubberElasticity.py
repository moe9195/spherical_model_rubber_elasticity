from dataclasses import dataclass
from enum import Enum
from math import *
from typing import Tuple, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class DeformationType(Enum):
    UNIAXIAL = 1
    EQUIBIAXIAL = 2
    PURE_SHEAR = 3


class DistributionType(Enum):
    UNIFORM_GRID = 1
    OCTAHEDRON = 2


@dataclass
class NetworkParameters:
    n: int  # average number of links per chain
    N: int  # chain density per unit sphere
    G: float  # shear modulus
    h: float  # step size for numerical differentiation
    tolerance: float  # maximum percent tolerance before chain breaks
    std: float  # standard deviation for chain length distribution


class SphereTriangulator:
    @staticmethod
    def sphere_triangulation_grid(n: int, m: int) -> NDArray:
        theta = np.linspace(0, 2 * pi, num=n)
        phi = np.linspace(0, pi, num=m)
        M = np.zeros((3, 3, (n - 1) * (m - 1) * 2))
        ind = 0
        for i in range(n - 1):
            for j in range(m - 1, 0, -1):
                x1u = cos(theta[i]) * sin(phi[j])
                y1u = sin(theta[i]) * sin(phi[j])
                z1u = cos(phi[j])

                x2u = cos(theta[i + 1]) * sin(phi[j])
                y2u = sin(theta[i + 1]) * sin(phi[j])
                z2u = cos(phi[j])

                x3u = cos(theta[i]) * sin(phi[j - 1])
                y3u = sin(theta[i]) * sin(phi[j - 1])
                z3u = cos(phi[j - 1])

                x1l, y1l, z1l = x2u, y2u, z2u
                x2l, y2l, z2l = x3u, y3u, z3u

                x3l = cos(theta[i + 1]) * sin(phi[j - 1])
                y3l = sin(theta[i + 1]) * sin(phi[j - 1])
                z3l = cos(phi[j - 1])

                M[:, :, 2 * ind] = np.array([[x1u, x2u, x3u], [y1u, y2u, y3u], [z1u, z2u, z3u]])
                M[:, :, 2 * ind + 1] = np.array([[x1l, x2l, x3l], [y1l, y2l, y3l], [z1l, z2l, z3l]])
                ind += 1
        return M

    @staticmethod
    def sphere_triangulation_octahedron(iterations: int, radius: float) -> NDArray:
        A = np.array([1, 0, 0])
        B = np.array([0, 1, 0])
        C = np.array([0, 0, 1])
        triangles = np.vstack((A, B, C, A, B, -C, -A, B, C, -A, B, -C, -A, -B, C, -A, -B, -C, A, -B, C, A, -B, -C))
        selector = np.arange(1, triangles.shape[0] - 1, 3)
        Ap = triangles[selector - 1]
        Bp = triangles[selector]
        Cp = triangles[selector + 1]

        for _ in range(1, iterations + 1):
            AB_2 = GeometryUtils.arsunit((Ap + Bp) / 2, radius)
            AC_2 = GeometryUtils.arsunit((Ap + Cp) / 2, radius)
            CB_2 = GeometryUtils.arsunit((Cp + Bp) / 2, radius)
            Ap = np.vstack((Ap, AB_2, AC_2, AC_2))
            Bp = np.vstack((AB_2, Bp, AB_2, CB_2))
            Cp = np.vstack((AC_2, CB_2, CB_2, Cp))

        M = np.array([Ap, Bp, Cp])
        return np.transpose(M, (2, 0, 1))


class GeometryUtils:
    @staticmethod
    def arsnorm(A: NDArray) -> NDArray:
        return np.sum(np.abs(A) ** 2, axis=0) ** (1. / 2)

    @staticmethod
    def arsunit(A: NDArray, radius: float) -> NDArray:
        normOfA = GeometryUtils.arsnorm(A.transpose())
        return radius * (np.divide(A, np.transpose(np.vstack((normOfA, normOfA, normOfA)))))

    @staticmethod
    def det3(a: NDArray) -> float:
        return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
                - a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
                + a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))


class DeformationCalculator:
    @staticmethod
    def deform(lam: NDArray, deform_type: DeformationType) -> NDArray:
        if deform_type == DeformationType.UNIAXIAL:
            lam1 = lam
            lam2 = np.reciprocal(np.sqrt(lam1))
            lam3 = lam2
        elif deform_type == DeformationType.EQUIBIAXIAL:
            lam1 = lam
            lam2 = lam
            lam3 = np.reciprocal(lam ** 2)
        elif deform_type == DeformationType.PURE_SHEAR:
            lam1 = lam
            lam2 = np.ones((1, lam.size))
            lam3 = np.reciprocal(lam1)
        else:
            raise ValueError("Invalid deformation type")
        return np.vstack((lam3, lam2, lam1))


class ChainNetwork:
    @staticmethod
    def inverse_langevin(x: NDArray) -> NDArray:
        return (x < 0.99) * x * (3 - x ** 2) / (1 - x ** 2)

    @staticmethod
    def solid_angle(M: NDArray) -> NDArray:
        s = np.shape(M)
        omega = np.zeros([1, s[2]])
        for i in range(s[2]):
            X = M[:, :, i]
            R1, R2, R3 = X[:, 0], X[:, 1], X[:, 2]
            N = GeometryUtils.arsnorm(X)
            n1, n2, n3 = N[0], N[1], N[2]
            D = abs(GeometryUtils.det3(X))
            A = D / ((n1 * n2 * n3) + ((np.dot(R1, R2) * n3) + (np.dot(R1, R3) * n2) + (np.dot(R2, R3) * n1)))
            omega[0, i] = atan(A) * 2
        return omega

    @staticmethod
    def spherical_triangle_centroid(M: NDArray) -> NDArray:
        s = M.shape
        C = np.zeros([3, s[2]])
        for i in range(s[2]):
            X = M[:, :, i]
            R1, R2, R3 = X[:, 0], X[:, 1], X[:, 2]
            temp = (R1 + R2 + R3) / 3
            temp = temp / np.linalg.norm(temp)
            C[:, i] = temp
        return C


class ForceCalculator:
    @staticmethod
    def force(r: NDArray, omega: NDArray, n0: NDArray, G: float, h: float) -> Tuple[NDArray, NDArray, NDArray]:
        b = ChainNetwork.inverse_langevin(r / np.sqrt(n0))
        s = (r > 0) * omega * (G * np.sqrt(n0) * ((r * b) + np.log(b / np.sinh(b)))) + (r <= 0) * 0
        f = np.diff(s, axis=0) / h
        f[np.isnan(f)] = 0
        F = (1 / (4 * pi)) * np.sum(f, 1)
        return s, f, F


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


class PlotManager:
    @staticmethod
    def plot_relation(lam0: NDArray, F: NDArray, Fnew: NDArray, fig: plt.Figure) -> None:
        plt.ylim([0, np.max(F) + 0.01])
        plt.margins(0, 0)
        plt.plot(lam0, F, 'k', lam0, Fnew, 'k')
        plt.grid()
        plt.xlabel('Stretch ratio (%)')
        plt.ylabel('Tensile force (N)')
        plt.title('Stress Strain Relation')
