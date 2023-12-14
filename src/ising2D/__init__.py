from __future__ import annotations

import numpy as np
import numpy.typing as typing
from numpy import random as rand


class ising2D:
    """
    The ising2D class allows users to set up a square grid of initially randomized
    spins representing a 2D Ising model. Note that the class changes state when
    simulation methods are called, so it will not remain random.
    It is initialized by two parameters:

    Inputs:
        grid_size: (int) The number of spin sites along each edge of the square grid
        temp: (float) The temperature of the system, in units of kT/J
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

    def metropolis(self) -> tuple[typing.NDArray[np.int64], int]:
        """
        Runs one iteration of the Metropolis simulation algorithm on the grid.
        Here, one iteration means that ``grid_size``:math:`^2` sites are randomly
        chosen and considered for a potential spin flip.

        Returns a tuple:
            1. (numpy array) The state of the grid after one Metropolis iteration
            2. (int) The sum of spin directions over the whole grid
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
        Runs ``num_iter`` iterations of the class instance's chosen algorithm
        (``self.algorithm``) sequentially and records the net magnetization at each
        iteration.

        Inputs:
            num_iter: (int) The number of iterations to run the simulation forward

        Returns a tuple:
            1. (numpy array) The state of the grid after `num_iter` simulation
            algorithm iterations

            2. (numpy array) A numpy array of length `num_iter` containing the net
            magnetization of the grid after each simulation algorithm iteration
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
