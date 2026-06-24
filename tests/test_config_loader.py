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


def test_repoConfigDisablesShapingAnneal():
    # Fix for SUICIDE1_NONCONVERGENCE (docs/observations.md): the shipped config
    # keeps PBRS shaping ON for the whole run (no anneal), so the late-reached
    # hard stages keep a dense gradient. PBRS is policy-invariant, so this does
    # not change the optimum. See docs/REWARD_LOG.md.
    cfg = loadConfig('config.yaml')
    assert cfg.reward.shapingAnneal == 'none'


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
    assert w.throttleResponse > 0
    assert not hasattr(w, 'maxThrust')      # old knob removed


def test_liftoffCapableValidation(tmp_path):
    # Peak thrust at full mass must beat gravity, else fail fast.
    with pytest.raises(ValueError):
        loadConfig(_writeConfig(
            tmp_path,
            'world: {maxThrustForce: 5.0, dryMass: 1.0, fuelMass: 0.6, gravity: 9.8}',
        ))


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


def test_engineModeFieldRemoved(tmp_path):
    # The analog world is gone — the engine is always the binary suicide burn.
    # engineMode is no longer a field, so a stray key fails fast (unknown kwarg).
    with pytest.raises(TypeError):
        loadConfig(_writeConfig(tmp_path, 'world: {engineMode: suicideBurn}'))


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


def test_worldHasGimbalResponseDefault():
    # Gimbal slew-rate limiter (command-units/s). Mirrors throttleResponse for the
    # engine spool: the nozzle eases toward the commanded angle instead of snapping.
    cfg = loadConfig('config.yaml')
    assert cfg.world.gimbalResponse == 4.0


def test_gimbalResponseChangesWorldHash():
    # gimbalResponse alters the rotational dynamics, so it is a hashed world field:
    # changing it must invalidate old checkpoints (retrain required).
    import dataclasses
    base = loadConfig('config.yaml')
    bumped = dataclasses.replace(base, world=dataclasses.replace(base.world, gimbalResponse=8.0))
    assert base.computeWorldHash() != bumped.computeWorldHash()


def test_gimbalResponseMustBePositive(tmp_path):
    p = tmp_path / 'config.yaml'
    p.write_text('world: {gimbalResponse: 0.0}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='gimbalResponse'):
        loadConfig(str(p))
