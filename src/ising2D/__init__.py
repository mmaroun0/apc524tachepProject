from __future__ import annotations

import numpy as np
import numpy.typing as typing
from numpy import random as rand


class ising2D:
    def __init__(self, grid_size: int, temp: float):
        self.grid_size = grid_size
        self.temperature = temp
        self.grid = rand.choice(
            [i for i in range(-1, 2) if i != 0], size=(grid_size, grid_size)
        )
        self.algorithm = "metropolis"

    def metropolis(self) -> tuple[typing.NDArray[np.int64], int]:
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
