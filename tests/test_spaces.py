# tests/test_spaces.py
import numpy as np
import pytest

from src.config.loader import loadConfig
from src.env.physics import BoosterState
from src.env.spaces import OBS_DIM, ACTION_DIM, VEL_REF, OMEGA_REF, encodeObs, toEnvAction


@pytest.fixture
def world():
    return loadConfig('config.yaml').world


def test_dims():
    assert OBS_DIM == 10
    assert ACTION_DIM == 2


def test_restingUprightStateEncodes(world):
    state = BoosterState(x=0.0, y=30.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=1.0, spool=0.0)
    obs = encodeObs(state, world)
    expected = np.array([0.0, 30.0 / world.ceiling, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert obs.dtype == np.float32
    assert obs.shape == (OBS_DIM,)
    np.testing.assert_allclose(obs, expected, atol=1e-6)


def test_spoolOccupiesLastIndex(world):
    state = BoosterState(y=30.0, fuel=0.5, spool=0.7)
    obs = encodeObs(state, world)
    assert obs[8] == pytest.approx(0.7)


def test_ignitionsRemainingAtIndexNine(world):
    fresh = encodeObs(BoosterState(y=30.0, engineTransitions=0), world)
    oneUsed = encodeObs(BoosterState(y=30.0, engineTransitions=1), world)
    locked = encodeObs(BoosterState(y=30.0, engineTransitions=2), world)
    assert fresh[9] == pytest.approx(1.0)
    assert oneUsed[9] == pytest.approx(0.5)
    assert locked[9] == pytest.approx(0.0)


def test_normalizationUsesRefs(world):
    state = BoosterState(
        x=world.width / 2, y=world.ceiling, vx=VEL_REF, vy=-VEL_REF,
        theta=0.5, omega=OMEGA_REF, fuel=0.25,
    )
    obs = encodeObs(state, world)
    assert obs[0] == pytest.approx(1.0)
    assert obs[1] == pytest.approx(1.0)
    assert obs[2] == pytest.approx(1.0)
    assert obs[3] == pytest.approx(-1.0)
    assert obs[4] == pytest.approx(np.sin(0.5), abs=1e-6)
    assert obs[5] == pytest.approx(np.cos(0.5), abs=1e-6)
    assert obs[6] == pytest.approx(1.0)
    assert obs[7] == pytest.approx(0.25)


def test_toEnvActionMapsTanhSpaceToEnvSpace():
    np.testing.assert_allclose(toEnvAction(np.array([-1.0, -1.0])), [0.0, -1.0])
    np.testing.assert_allclose(toEnvAction(np.array([1.0, 1.0])), [1.0, 1.0])
    np.testing.assert_allclose(toEnvAction(np.array([0.0, 0.0])), [0.5, 0.0])


def test_toEnvActionPreservesBatchShape():
    batch = np.zeros((5, ACTION_DIM), dtype=np.float32)
    out = toEnvAction(batch)
    assert out.shape == (5, ACTION_DIM)
    np.testing.assert_allclose(out[:, 0], 0.5)


def test_inWorldStatesStayBounded(world):
    # Any state inside the world box with sane velocities encodes within [-1.5, 1.5].
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = BoosterState(
            x=rng.uniform(-world.width / 2, world.width / 2),
            y=rng.uniform(0, world.ceiling),
            vx=rng.uniform(-VEL_REF, VEL_REF),
            vy=rng.uniform(-VEL_REF, VEL_REF),
            theta=rng.uniform(-np.pi, np.pi),
            omega=rng.uniform(-OMEGA_REF, OMEGA_REF),
            fuel=rng.uniform(0, 1),
        )
        obs = encodeObs(state, world)
        assert np.all(np.abs(obs) <= 1.5)
