import numpy as np
from numpy.typing import NDArray

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