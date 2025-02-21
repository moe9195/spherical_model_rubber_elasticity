from math import pi
import numpy as np
from numpy.typing import NDArray
from ..geometry.geometry_utils import GeometryUtils

class SphereTriangulator:
    @staticmethod
    def sphere_triangulation_grid(n: int, m: int) -> NDArray:
        theta = np.linspace(0, 2 * pi, num=n)
        phi = np.linspace(0, pi, num=m)
        M = np.zeros((3, 3, (n - 1) * (m - 1) * 2))
        ind = 0
        for i in range(n - 1):
            for j in range(m - 1, 0, -1):
                x1u = np.cos(theta[i]) * np.sin(phi[j])
                y1u = np.sin(theta[i]) * np.sin(phi[j])
                z1u = np.cos(phi[j])

                x2u = np.cos(theta[i + 1]) * np.sin(phi[j])
                y2u = np.sin(theta[i + 1]) * np.sin(phi[j])
                z2u = np.cos(phi[j])

                x3u = np.cos(theta[i]) * np.sin(phi[j - 1])
                y3u = np.sin(theta[i]) * np.sin(phi[j - 1])
                z3u = np.cos(phi[j - 1])

                x1l, y1l, z1l = x2u, y2u, z2u
                x2l, y2l, z2l = x3u, y3u, z3u

                x3l = np.cos(theta[i + 1]) * np.sin(phi[j - 1])
                y3l = np.sin(theta[i + 1]) * np.sin(phi[j - 1])
                z3l = np.cos(phi[j - 1])

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