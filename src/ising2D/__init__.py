from __future__ import annotations

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as typing
from matplotlib.figure import Figure
from numpy import random as rand


@numba.jit(
    numba.int64[:, ::1](numba.int64[:, ::1], numba.int64, numba.float32, numba.boolean),
    nopython=True,
)  # type: ignore
def metropolis_grid(
    grid: typing.NDArray[np.int64], grid_size: int, temperature: float, test: bool
) -> typing.NDArray[np.int64]:
    """
    Perform ``grid_size``:math:`^2` potential spin flips, where the choice of sites
    and decision of whether or not to flip is based on the Metropolis algorithm. This
    function is accelerated with numba's just-in-time compiler for performance.

    Performs Metropolis Algorithm:
        Calculates a change in energy: delta_e.
        If lower than 0, the state changes to the new energy.
        If greater than 0, the state is changed with the probability of
            e^-(delta_e/temperature).

    Parameters:
        1. grid (NDArray[int]): 2D array of energy states representing the system
        2. grid_size (int): Length of one side of the grid
        3. temperature (float): Temperature of the system in k*T/J
        4. test (bool): Marks if function is running in pytest - users should always set
        to False

    Returns:
        1. grid (NDArray[int]): 2D array of states representing the system
        post-simulation
    """
    if test:
        rand.seed(0)
    for _ndx in range(grid_size**2):
        flip_ndx_row = rand.randint(0, grid_size)
        flip_ndx_col = rand.randint(0, grid_size)
        delta_e = (
            2
            * grid[flip_ndx_row, flip_ndx_col]
            * (
                grid[(flip_ndx_row + 1) % grid_size, flip_ndx_col]
                + grid[flip_ndx_row, (flip_ndx_col + 1) % grid_size]
                + grid[(flip_ndx_row - 1) % grid_size, flip_ndx_col]
                + grid[flip_ndx_row, (flip_ndx_col - 1) % grid_size]
            )
        )

        the_seed = rand.rand()
        if delta_e < 0 or the_seed < np.exp(-delta_e / temperature):
            grid[flip_ndx_row, flip_ndx_col] = -grid[flip_ndx_row, flip_ndx_col]
    return grid


class ising2D:
    """
    The ising2D class allows users to create a square grid of initially randomized
    spins representing a 2D Ising system. Note that the class changes states when
    simulation methods are called, brining the system to equilibrium under enough
    iterations. Plotting of the net magnetization and grid are also included.
    It is initialized by two parameters:

    Inputs:
        1. grid_size (int): The number of spin sites along each edge of the square grid
        2. temp (float): The temperature of the system, in units of kT/J
        3. algorithm (str): The algorithm to use. Options are ["metropolis"]
    """

    def __init__(self, grid_size: int, temp: float, algorithm: str):
        self.grid_size = grid_size
        self.temperature = temp
        self.grid = rand.choice(
            [i for i in range(-1, 2) if i != 0], size=(grid_size, grid_size)
        )
        if algorithm not in ["metropolis"]:
            raise Exception(f"Incompatible algorithm {algorithm}. Exiting")
        else:
            self.algorithm = algorithm

    def metropolis(self, test: bool = False) -> tuple[typing.NDArray[np.int64], int]:
        """
        Runs one iteration of the Metropolis simulation algorithm on the grid.
        Here, one iteration means that ``grid_size``:math:`^2` sites are randomly
        chosen and considered for a potential spin flip.

        Parameters:
            1. test (bool): Marks if function is running in pytest - users should always
            set to False

        Returns a tuple:
            1. grid (numpy array): The state of the grid after one Metropolis iteration
            2. gross_mags (int): The sum of spin directions over the whole grid
        """
        self.grid = metropolis_grid(self.grid, self.grid_size, self.temperature, test)
        gross_mags = int(sum(sum(self.grid)))
        return self.grid, gross_mags

    def alg_sweep(
        self, num_iter: int, test: bool = False
    ) -> tuple[typing.NDArray[np.int64], typing.NDArray[np.float64]]:
        """
        Runs ``num_iter`` iterations of the class instance's chosen algorithm
        (``self.algorithm``) sequentially and records the net magnetization at each
        iteration.

        Inputs:
            1. num_iter (int): The number of iterations to run the simulation
            2. test (bool): Marks if function is running in pytest - users should always
            set to False

        Returns a tuple:
            1. grid (numpy array) The state of the grid after `num_iter` simulation
            algorithm iterations
            2. net_mags (numpy array) A numpy array of length `num_iter` containing the
            net magnetization of the grid after each simulation algorithm iteration
        """
        net_mags = np.zeros(
            [
                num_iter,
            ],
            dtype=float,
        )
        if self.algorithm == "metropolis":
            for ndx in range(num_iter):
                self.grid, gross_mags = self.metropolis(test)
                net_mags[ndx] = gross_mags / (self.grid_size) ** 2
        return self.grid, net_mags

    def plot_net_mags(
        self,
        net_mags: typing.NDArray[np.int64],
        filename: str,
    ) -> Figure:
        """
        Creates 1D plot of net magnetization per time.

        Parameters:
            1. num_iter (int): Total number of times the algorithm repeated.
            2. net_mags (NDArray[float]): Array of magnetization of the entire system
            for each time.
            3. filename (string): name of file, must include file type extension.
            4. title (string): title of plot located at top center of figure.
        """
        if filename == "":
            raise Exception("Filename required. Exiting")
        else:
            numIter = np.size(net_mags)
            t = np.linspace(0, numIter, numIter)
            fig = plt.figure()
            plt.plot(t, net_mags, figure=fig)
            plt.xlabel("Time [MC Sweeps]", figure=fig)
            plt.ylabel("Magnitude", figure=fig)
            plt.title(
                "Net Magnetization vs. Time of 2D Ising Model Simulation", figure=fig
            )
            fig.savefig(filename)
            return fig

    def plot_grid(self, filename: str) -> Figure:
        """
        Creates 2D plot of system grid.

        Parameters:
            1. filename (string): name of file, must include file type extension.
            2. title (string): title of plot located at top center of figure.
        """
        if filename == "":
            raise Exception("Filename required. Exiting")
        else:
            fig = plt.figure()
            plt.imshow(self.grid, figure=fig)
            plt.title("Simulation of 2d Ising Model", figure=fig)
            plt.colorbar()
            fig.savefig(filename)
            return fig
