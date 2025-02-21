from math import pi
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from .chain_network import ChainNetwork

class ForceCalculator:
    @staticmethod
    def force(r: NDArray, omega: NDArray, n0: NDArray, G: float, h: float) -> Tuple[NDArray, NDArray, NDArray]:
        b = ChainNetwork.inverse_langevin(r / np.sqrt(n0))
        s = (r > 0) * omega * (G * np.sqrt(n0) * ((r * b) + np.log(b / np.sinh(b)))) + (r <= 0) * 0
        f = np.diff(s, axis=0) / h
        f[np.isnan(f)] = 0
        F = (1 / (4 * pi)) * np.sum(f, 1)
        return s, f, F