import numpy as np
from numpy import random as rand
from matplotlib import pyplot as plt
import dataclasses

class ising2D:
    def __init__(self, grid_size: int, temp: float):
        self.grid_size = grid_size
        self.temperature = temp
        self.grid = rand.choice([i for i in range(-1,2) if i != 0],size=(grid_size,grid_size))
        self.algorithm = 'metropolis'

    def met2DIsing(self):
        mm = np.size(self.grid,0)
        for ndx in range(mm**2):
            flip_ndx_row = rand.randint(0,np.size(self.grid,0))
            flip_ndx_col = rand.randint(0,np.size(self.grid,1))
            delta_e = 2*self.grid[flip_ndx_row,flip_ndx_col]*(self.grid[(flip_ndx_row+1)%mm,flip_ndx_col]+\
                                                        self.grid[flip_ndx_row,(flip_ndx_col+1)%mm]+\
                                                        self.grid[(flip_ndx_row-1)%mm,flip_ndx_col]+\
                                                        self.grid[flip_ndx_row,(flip_ndx_col-1)%mm])
            the_seed = rand.rand()
            if delta_e < 0 or the_seed < np.exp(-delta_e/self.temperature):
                self.grid[flip_ndx_row,flip_ndx_col] = -self.grid[flip_ndx_row,flip_ndx_col]
            gross_mags = int(np.sum(np.sum(self.grid)))
        return self.grid, gross_mags