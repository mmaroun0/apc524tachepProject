import ising2D
import numpy as np
import pytest
from numpy import random as rand


@pytest.fixture
def true_grid_seed_0():
    """
    Baseline for ising2D

    pytest fixture that is the correct 3x3 grid when np.rand.seed is set to 0
    before creating the ising2D object with grid_size = 3 and temp = 5.0.

    Notice that if the number of rand calls changes in ising2D's __init__, a
    new baseline true_grid_seed_0 may need to be created.
    """
    return np.array([[-1, 1, 1], [-1, 1, 1], [1, 1, 1]])


@pytest.fixture
def true_next_grid_seed_0():
    """
    Baseline for ising2D

    The correct grid that results from calling met2DIsing on true_grid_seed_0
    when rand.seed was set to 0 just before calling met2DIsing. This may also
    need to be changed depending on when rand is called in the source code.
    """
    return np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])


@pytest.fixture
def true_grid_15_met_sweeps_seed_0():
    """
    Baseline for ising2D

    The correct grid that results from calling met2DIsing on true_grid_seed_0
    when rand.seed is set to 0 before calling alg_sweep for 15 iterations. This
    baseline may need to be changed depending on when rand is called in the source
    code.
    """
    return np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]])


@pytest.fixture
def true_net_mags_15_met_sweeps_seed_0():
    """
    Baseline for ising2D

    The correct net_mags history array that results from calling met2DIsing on
    true_grid_seed_0 when rand.seed is set to 0 before calling alg_sweep for 15
    iterations. This baseline may need to be changed depending on when rand is
    called in the source code.
    """
    return np.array(
        [
            0.33333333,
            0.55555556,
            0.33333333,
            0.55555556,
            0.33333333,
            0.55555556,
            0.33333333,
            0.55555556,
            0.33333333,
            0.55555556,
            0.33333333,
            0.55555556,
            0.33333333,
            0.55555556,
            0.33333333,
        ]
    )


def test_init(true_grid_seed_0):
    """
    Test for ising2D

    Test if the initial values are set properly. Notice that if the number of
    rand calls changes in __init__, a new baseline true_grid_seed_0 may need to
    be generated.
    """
    grid_size = 3
    temp = 5.0
    algorithm = "metropolis"
    rand.seed(0)

    model = ising2D.ising2D(grid_size, temp, algorithm)
    assert grid_size == model.grid_size
    assert model.temperature == pytest.approx(temp)
    assert (model.grid == true_grid_seed_0).all()


def test_unknown_algorithm():
    """
    Test for ising2D

    Test that if the Monte Carlo algorithm specified for the class is not in the
    list of known algorithms, then the proper Exception is raised upon trying to
    instantiate the class.
    """
    grid_size = 3
    temp = 5.0
    algorithm = "foo"
    rand.seed(0)

    with pytest.raises(Exception) as excinfo:
        ising2D.ising2D(grid_size, temp, algorithm)
    assert "Incompatible algorithm" in str(excinfo.value)


def test_metropolis(true_next_grid_seed_0):
    """
    Test for ising2D

    In this test, one of the values in the grid (at [1,2]) should flip.
    Please note that this depends on how many times rand is called, so if that
    changes, a new baseline true_grid_seed_0 needs to be made.
    """
    grid_size = 3
    temp = 5.0
    algorithm = "metropolis"
    rand.seed(0)

    model = ising2D.ising2D(grid_size, temp, algorithm)
    rand.seed(0)
    next_grid, next_gross_mags = model.metropolis(test=True)
    assert (next_grid == true_next_grid_seed_0).all()
    assert next_gross_mags == np.sum(np.sum(true_next_grid_seed_0))


def test_metropolis_sweep(
    true_grid_15_met_sweeps_seed_0, true_net_mags_15_met_sweeps_seed_0
):
    """
    Test for ising2D

    In this test, the final grid and net magnetization history after 15 MC sweeps on
    the metropolis algorithm are tested against a baseline when rand.seed(0) is set.
    This test depends on the number of times rand() is called, so if that changes, new
    baseline true_grid_15_met_sweeps_seed_0 and true_net_mags_15_met_sweeps_seed_0
    need to be made.
    """
    grid_size = 3
    temp = 5.0
    algorithm = "metropolis"
    num_iter = 15
    rand.seed(0)

    model = ising2D.ising2D(grid_size, temp, algorithm)
    new_grid, net_mags = model.alg_sweep(num_iter, test=True)
    assert (new_grid == true_grid_15_met_sweeps_seed_0).all()
    assert net_mags == pytest.approx(true_net_mags_15_met_sweeps_seed_0)
