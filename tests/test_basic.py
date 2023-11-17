import ising2D
import numpy as np
from numpy import random as rand

import pytest

@pytest.fixture
def true_grid_seed_0():
    '''
    Baseline for ising2D

    pytest fixture that is the correct 3x3 grid when np.rand.seed is set to 0
    before creating the ising2D object with grid_size = 3 and temp = 5.0.

    Notice that if the number of rand calls changes in ising2D's __init__, a
    new baseline true_grid_seed_0 may need to be created.
    '''
    return np.array([[-1,1,1],[-1,1,1],[1,1,1]])

@pytest.fixture
def true_next_grid_seed_0():
    '''
    Baseline for ising2D

    The correct grid that results from calling met2DIsing on true_grid_seed_0
    when rand.seed was set to 0 just before calling met2DIsing. This may also
    need to be changed depending on when rand is called in the source code.
    '''
    return np.array([[1,1,1],[-1,1,1],[-1,-1,1]])

def test_init(true_grid_seed_0):
    """
    Test for ising2D

    Test if the initial values are set properly. Notice that if the number of
    rand calls changes in __init__, a new baseline true_grid_seed_0 may need to
    be generated.
    """
    grid_size = 3
    temp = 5.0
    rand.seed(0)

    model = ising2D.ising2D(grid_size,temp)
    assert grid_size == model.grid_size
    assert model.temperature == pytest.approx(temp)
    assert (model.grid == true_grid_seed_0).all()

def test_metropolis(true_next_grid_seed_0):
    """
    Test for ising2D

    In this test, one of the values in the grid (at [1,2]) should flip.
    Please note that this depends on how many times rand is called, so if that
    changes, a new baseline true_grid_seed_0 needs to be made.
    """
    grid_size = 3
    temp = 5.0
    rand.seed(0)

    model = ising2D.ising2D(grid_size,temp)
    rand.seed(0)
    next_grid, next_gross_mags = model.met2DIsing()
    assert (next_grid == true_next_grid_seed_0).all()
    assert next_gross_mags == np.sum(np.sum(true_next_grid_seed_0))