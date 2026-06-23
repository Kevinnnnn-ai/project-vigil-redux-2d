# tests/test_config_loader.py
import textwrap

import pytest

from src.config.loader import Config, loadConfig


def _writeConfig(tmpPath, body):
    """Write a YAML snippet to a temp file and return its path as a string.
    Omitted sections fall back to dataclass defaults, so snippets stay tiny."""
    path = tmpPath / 'config.yaml'
    path.write_text(textwrap.dedent(body), encoding='utf-8')
    return str(path)


def test_loadsRepoConfigYaml():
    # The real control panel at the repo root must parse into a Config.
    cfg = loadConfig('config.yaml')
    assert isinstance(cfg, Config)
    assert cfg.mode == 'train'
    assert cfg.world.gravity == pytest.approx(9.8)


def test_defaultsFillMissingSections(tmp_path):
    path = _writeConfig(tmp_path, '''
        world:
          gravity: 3.7
    ''')
    cfg = loadConfig(path)
    assert cfg.world.gravity == 3.7
    assert cfg.world.maxThrustForce == 30.0          # untouched default preserved
    assert cfg.training.lr == pytest.approx(3.0e-4)  # whole section defaulted


def test_listFieldsBecomeTuples(tmp_path):
    path = _writeConfig(tmp_path, '''
        training:
          evalSeeds: [0, 1, 2, 3]
          hidden: [128, 128]
    ''')
    cfg = loadConfig(path)
    assert cfg.training.evalSeeds == (0, 1, 2, 3)
    assert cfg.training.hidden == (128, 128)


def test_curriculumStagesParse(tmp_path):
    path = _writeConfig(tmp_path, '''
        curriculum:
          promoteAt: 0.9
          stages:
            - name: easy
              altitude: [5.0, 8.0]
              xOffset: [-1.0, 1.0]
              vx: [-0.1, 0.1]
              vy: [-1.0, -0.5]
              tilt: [-0.02, 0.02]
              omega: [-0.01, 0.01]
    ''')
    cfg = loadConfig(path)
    assert cfg.curriculum.promoteAt == 0.9
    assert len(cfg.curriculum.stages) == 1
    assert cfg.curriculum.stages[0].name == 'easy'
    assert cfg.curriculum.stages[0].altitude == (5.0, 8.0)


def test_repoCurriculumLadderEndsAtFull():
    cfg = loadConfig('config.yaml')
    assert len(cfg.curriculum.stages) == 5
    assert cfg.curriculum.stages[0].name == 'touchdown'   # the samplable first rung
    assert cfg.curriculum.stages[-1].name == 'full'


def test_worldHashStableForSameWorld(tmp_path):
    a = loadConfig(_writeConfig(tmp_path, 'world: {dt: 0.05}'))
    b = loadConfig(_writeConfig(tmp_path, 'world: {dt: 0.05}'))
    assert a.computeWorldHash() == b.computeWorldHash()


def test_changingGravityChangesWorldHash(tmp_path):
    base = loadConfig(_writeConfig(tmp_path, 'world: {gravity: 9.8}'))
    other = loadConfig(_writeConfig(tmp_path, 'world: {gravity: 3.7}'))
    assert base.computeWorldHash() != other.computeWorldHash()


def test_physicsModelVersionParticipatesInWorldHash(tmp_path, monkeypatch):
    # The physics-MODEL version is folded into the world hash, so changing the
    # simulation model itself (not just config fields) invalidates old
    # checkpoints even when every world: field is identical. Guards against a
    # future hash refactor silently dropping the version tag.
    import src.config.loader as loader
    cfg = loadConfig(_writeConfig(tmp_path, 'world: {dt: 0.05}'))
    before = cfg.computeWorldHash()
    monkeypatch.setattr(loader, 'PHYSICS_MODEL_VERSION', 'some-other-model')
    assert cfg.computeWorldHash() != before


def test_rewardChangeDoesNotChangeWorldHash(tmp_path):
    base = loadConfig(_writeConfig(tmp_path, 'reward: {shapingCoef: 1.0}'))
    other = loadConfig(_writeConfig(tmp_path, 'reward: {shapingCoef: 0.2}'))
    assert base.computeWorldHash() == other.computeWorldHash()


def test_curriculumChangeDoesNotChangeWorldHash(tmp_path):
    # Spawn ranges change what the agent practices, not the world it acts in.
    base = loadConfig(_writeConfig(tmp_path, 'curriculum: {promoteAt: 0.8}'))
    other = loadConfig(_writeConfig(tmp_path, 'curriculum: {promoteAt: 0.5}'))
    assert base.computeWorldHash() == other.computeWorldHash()


def test_nonPositiveDtRaises(tmp_path):
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(tmp_path, 'world: {dt: 0.0}'))


def test_worldHasMassAndSpoolFields():
    cfg = loadConfig('config.yaml')
    w = cfg.world
    assert w.dryMass > 0 and w.fuelMass > 0
    assert w.maxThrustForce > 0
    assert 0.0 < w.minThrottle < 1.0
    assert w.throttleResponse > 0
    assert not hasattr(w, 'maxThrust')      # old knob removed


def test_liftoffCapableValidation(tmp_path):
    # Peak thrust at full mass must beat gravity, else fail fast.
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(
            tmp_path,
            'world: {maxThrustForce: 5.0, dryMass: 1.0, fuelMass: 0.6, gravity: 9.8}',
        ))


def test_minThrottleRangeValidation(tmp_path):
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(tmp_path, 'world: {minThrottle: 1.5}'))


def test_invalidModeRaises(tmp_path):
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(tmp_path, 'mode: spectate'))


def test_emptyCurriculumRaises(tmp_path):
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(tmp_path, 'curriculum: {stages: []}'))


def test_invertedStageRangeRaises(tmp_path):
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(tmp_path, '''
            curriculum:
              stages:
                - name: bad
                  altitude: [12.0, 8.0]
                  xOffset: [-1.0, 1.0]
                  vx: [-0.1, 0.1]
                  vy: [-1.0, -0.5]
                  tilt: [-0.02, 0.02]
                  omega: [-0.01, 0.01]
        '''))


def test_repoConfigPassesValidation():
    loadConfig('config.yaml')


def test_runtimeModelDefaultsToLux(tmp_path):
    # no runtime section — dataclass default applies
    path = _writeConfig(tmp_path, 'mode: train')
    cfg = loadConfig(path)
    assert cfg.runtime.model == 'lux'


def test_runtimeModelDoesNotAffectWorldHash(tmp_path):
    # runtime.model is a selector, never part of the compatibility hash
    lux = loadConfig(_writeConfig(tmp_path, '''
        world:
          width: 40.0
        runtime:
          model: lux
    '''))
    solis = loadConfig(_writeConfig(tmp_path, '''
        world:
          width: 40.0
        runtime:
          model: solis
    '''))
    assert lux.computeWorldHash() == solis.computeWorldHash()


# <agent_guardrail>
#   [CRITICAL]: lux (analog) and solis (suicideBurn) are INTENTIONALLY DIFFERENT
#               worlds — different engine dynamics -> different hash -> checkpoints
#               are NOT interchangeable (M0 contract enforces this correctly).
#   [VALIDATION]: test_luxAndSolisShipWithDifferentWorldHash reads the REAL repo
#                 configs via relative paths — pytest must be invoked from repo root
#                 (enforced by pytest.ini rootdir). If this test ever fails, one
#                 config's engineMode was changed without updating the other.
# </agent_guardrail>
def test_luxAndSolisShipWithDifferentWorldHash():
    # lux (analog) and solis (suicideBurn) are intentionally DIFFERENT worlds now:
    # different engine dynamics -> different hash -> checkpoints are NOT
    # interchangeable (a suicideBurn net must never load in an analog world).
    lux = loadConfig('configs/lux/baseline.yaml')
    solis = loadConfig('configs/solis/baseline.yaml')
    assert lux.world.engineMode == 'analog'
    assert solis.world.engineMode == 'suicideBurn'
    assert lux.computeWorldHash() != solis.computeWorldHash()


# <agent_context>
#   [ARCH]: engineMode selects between Lux's analog (continuous) throttle and
#           Solis's binary suicide-burn (full throttle or off). Must be a world
#           field so it is included in computeWorldHash() automatically via asdict().
#   [GOTCHA]: The hash-difference test relies on asdict(world) picking up the new
#             field — no manual hash wiring needed. If the field moves outside
#             WorldConfig, the hash contract breaks silently.
# </agent_context>
def test_engineModeDefaultsToAnalog(tmp_path):
    path = _writeConfig(tmp_path, 'mode: train')
    cfg = loadConfig(path)
    assert cfg.world.engineMode == 'analog'


def test_engineModeChangesWorldHash(tmp_path):
    analogPath = tmp_path / 'analog.yaml'
    analogPath.write_text(textwrap.dedent('''
        world:
          engineMode: analog
    '''), encoding='utf-8')
    suicidePath = tmp_path / 'suicide.yaml'
    suicidePath.write_text(textwrap.dedent('''
        world:
          engineMode: suicideBurn
    '''), encoding='utf-8')
    analog = loadConfig(str(analogPath))
    suicide = loadConfig(str(suicidePath))
    assert analog.computeWorldHash() != suicide.computeWorldHash()


def test_invalidEngineModeRaises(tmp_path):
    with pytest.raises(ValueError, match='engineMode'):
        loadConfig(_writeConfig(tmp_path, '''
            world:
              engineMode: turbo
        '''))


def test_legDropAndSettleStepCapDefaults():
    cfg = loadConfig('config.yaml')
    assert cfg.world.legDrop == 0.9
    assert cfg.world.settleStepCap == 120


def test_legDropChangesWorldHash():
    import dataclasses
    base = loadConfig('config.yaml')
    bumped = dataclasses.replace(base, world=dataclasses.replace(base.world, legDrop=1.2))
    assert base.computeWorldHash() != bumped.computeWorldHash()


def test_settleStepCapChangesWorldHash():
    import dataclasses
    base = loadConfig('config.yaml')
    bumped = dataclasses.replace(base, world=dataclasses.replace(base.world, settleStepCap=200))
    assert base.computeWorldHash() != bumped.computeWorldHash()


def test_legDropMustBePositive(tmp_path):
    p = tmp_path / 'config.yaml'
    p.write_text('world: {legDrop: 0.0}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='legDrop'):
        loadConfig(str(p))


def test_settleStepCapMustBePositiveInt(tmp_path):
    p = tmp_path / 'config.yaml'
    p.write_text('world: {settleStepCap: 0}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='settleStepCap'):
        loadConfig(str(p))
