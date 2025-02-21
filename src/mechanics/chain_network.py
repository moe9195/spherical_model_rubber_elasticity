from math import atan
import numpy as np
from numpy.typing import NDArray
from ..geometry.geometry_utils import GeometryUtils

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