import numpy as np
from numpy.typing import NDArray
from ..enums.deformation_type import DeformationType

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