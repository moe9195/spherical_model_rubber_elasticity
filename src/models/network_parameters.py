from dataclasses import dataclass

@dataclass
class NetworkParameters:
    n: int  # average number of links per chain
    N: int  # chain density per unit sphere
    G: float  # shear modulus
    h: float  # step size for numerical differentiation
    tolerance: float  # maximum percent tolerance before chain breaks
    std: float  # standard deviation for chain length distribution