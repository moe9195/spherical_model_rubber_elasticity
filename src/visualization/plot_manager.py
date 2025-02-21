import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

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