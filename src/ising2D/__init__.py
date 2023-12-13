from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as typing
from numpy import random as rand


class ising2D:
    """
    Ising2D Class creates a grid of randomly assigned spins representing the system
    and allows for a simulation algorithm to bring the grid into equilibrium.
    Plotting of the net magnetization and grid are also included.

    Parameters:
        grid_size (int): Length of one side of the grid.
        temp (float): Temperature of the system in k*T/J
    """

    def __init__(self, grid_size: int, temp: float):
        self.grid_size = grid_size
        self.temperature = temp
        self.grid = rand.choice(
            [i for i in range(-1, 2) if i != 0], size=(grid_size, grid_size)
        )
        self.algorithm = "metropolis"

    def metropolis(self) -> tuple[typing.NDArray[np.int64], int]:
        """
        Performs Metropolis Algorithm
        Calculates a change in energy: delta_e.
        If lower than 0, the state changes to the new energy.
        If greater than 0, the state is changed with the probability of
            e^-(delta_e/temperature).

        Returns:
            grid (NDArray[int]): 2D array of energy states representing the system.
            gross_mags (int): total magnetization.
        """
        for _ndx in range(self.grid_size**2):
            flip_ndx_row = rand.randint(0, np.size(self.grid, 0))
            flip_ndx_col = rand.randint(0, np.size(self.grid, 1))
            delta_e = (
                2
                * self.grid[flip_ndx_row, flip_ndx_col]
                * (
                    self.grid[(flip_ndx_row + 1) % self.grid_size, flip_ndx_col]
                    + self.grid[flip_ndx_row, (flip_ndx_col + 1) % self.grid_size]
                    + self.grid[(flip_ndx_row - 1) % self.grid_size, flip_ndx_col]
                    + self.grid[flip_ndx_row, (flip_ndx_col - 1) % self.grid_size]
                )
            )

            the_seed = rand.rand()
            if delta_e < 0 or the_seed < np.exp(-delta_e / self.temperature):
                self.grid[flip_ndx_row, flip_ndx_col] = -self.grid[
                    flip_ndx_row, flip_ndx_col
                ]
            gross_mags = int(np.sum(np.sum(self.grid)))
        return self.grid, gross_mags

    def alg_sweep(
        self, num_iter: int
    ) -> tuple[typing.NDArray[np.int64], typing.NDArray[np.float64]]:
        """
        Repeats algorithm num_iter times on the states.
        Allowing the system to reach a possible equilibrium state.

        Parameters:
            num_iter (int): Total number of times to repeat the algorithm.

        Returns:
            grid (NDArray[int]): 2D array of energy states representing the system.
            net_mags (NDArray[float]): Array of average magnetization of the entire
                system for each time.
        """
        net_mags = np.zeros(
            [
                num_iter,
            ],
            dtype=float,
        )
        if self.algorithm == "metropolis":
            for ndx in range(num_iter):
                self.grid, gross_mags = self.metropolis()
                net_mags[ndx] = gross_mags / (self.grid_size) ** 2
            return self.grid, net_mags
        else:
            raise Exception("Incompatible algorithm. Exiting")

    def plot_net_mags(
        self,
        num_iter: int,
        net_mags: typing.NDArray[np.int64],
        filename: str,
        title: str,
    ) -> None:
        """
        Creates 1D plot of net magnetization per time.

        Parameters:
            num_iter (int): Total number of times the algorithm repeated.
            net_mags (NDArray[float]): Array of magnetization of the entire system for
                each time.
            filename (string): name of file, must include file type extension.
            title (string): title of plot located at top center of figure.
        """
        if filename == "":
            raise Exception("Filename required. Exiting")
        else:
            numIter = np.size(net_mags)
            t = np.linspace(0, numIter, numIter)
            plt.plot(t, net_mags)
            plt.xlabel("Time [MC Sweeps]")
            plt.ylabel("Magnitude")
            plt.title(title)
            plt.savefig(filename)
            plt.close()

    def plot_grid(self, filename: str, title: str) -> None:
        """
        Creates 2D plot of system grid.

        Parameters:
            filename (string): name of file, must include file type extension.
            title (string): title of plot located at top center of figure.
        """
        if filename == "":
            raise Exception("Filename required. Exiting")
        else:
            plt.imshow(self.grid)
            plt.title(title)
            plt.colorbar()
            plt.savefig(filename)
            plt.close()
