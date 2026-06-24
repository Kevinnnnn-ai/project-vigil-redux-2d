# src/config/loader.py
# <agent_context>
#   [ARCH]: Single source of truth for runtime configuration. All knobs live in
#           config.yaml (camelCase keys). Dataclasses are frozen (immutable after
#           construction) so callers can cache the Config object safely.
#   [GOTCHA]: YAML keys map 1:1 to dataclass field names — both are camelCase.
#             Missing YAML sections fall back to dataclass field defaults entirely;
#             present sections are forwarded with **kwargs, so any unknown key in
#             config.yaml will raise TypeError at load time (fail-fast, by design).
#   [GOTCHA]: A bare section header with no body (e.g. `world:`) parses via
#             safe_load to {'world': None}. loadConfig() guards every section with
#             `or {}` so None collapses to an empty dict and dataclass defaults
#             apply — this is intentional, not a bug.
#   [GOTCHA]: list fields (evalSeeds, hidden, every curriculum range pair) must be
#             converted to tuple before constructing their frozen dataclass because
#             frozen dataclasses cannot hold mutable lists; loadConfig() and
#             _buildStage() do this conversion explicitly.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: computeWorldHash() must hash ONLY world fields. Never fold
#               reward/training/curriculum into it — that would wrongly block
#               watch/play across reward, training, or spawn-range edits, which
#               must stay compatible with existing models.
#   [CRITICAL]: Do NOT change the dataclass field names to snake_case; the YAML
#               control panel uses camelCase and the two must stay in sync.
#   [VALIDATION]: If this file is touched, run:
#                 python -m pytest tests/test_config_loader.py -v
# </agent_guardrail>
"""Single source of truth for runtime configuration.

Parses config.yaml into frozen dataclasses grouped by concern. The `world`
group is the compatibility boundary: computeWorldHash() hashes ONLY those
fields, so editing physics invalidates old models while editing
reward/training/curriculum never does.

YAML keys are camelCase and map 1:1 onto dataclass fields, so each section is
built with Cls(**section). Missing sections/keys fall back to the defaults here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

import yaml

# @CONFIG[physics-model-version]: folded into computeWorldHash so a change to the
# SIMULATION MODEL itself invalidates old checkpoints even when the world config
# fields are unchanged. 'pymunk-2' = the Pymunk (Chipmunk2D) rigid-body model with
# physically-collidable legs (src/env/physics.py BoosterSim), advanced in sub-ticks
# per env step so hard impacts resolve rigidly (no deep penetration / ooze-out).
# 'pymunk-1' was the same model before sub-stepping; the prior hand-written
# semi-implicit-Euler integrator was the implicit 'v0'. 'suicide-1' supersedes
# 'pymunk-2': the analog engine branch was removed, leaving the binary suicide-burn
# engine as the sole dynamics model — dynamics changed even though the world: fields
# are unchanged. Bump this string on any future physics-model change that alters
# dynamics without touching world fields.
PHYSICS_MODEL_VERSION = 'suicide-1'  # 'suicide-1' = the analog engine removed; the world is exclusively the binary suicide burn.


@dataclass(frozen=True)
class WorldConfig:
    width: float = 40.0
    ceiling: float = 60.0
    padWidth: float = 8.0
    gravity: float = 9.8
    dryMass: float = 1.0
    fuelMass: float = 0.6
    maxThrustForce: float = 30.0
    gimbalArm: float = 1.0
    momentInertiaCoef: float = 1.0
    maxGimbal: float = 0.35
    throttleResponse: float = 4.0
    gimbalResponse: float = 4.0   # nozzle slew rate (command-units/s); full -1->+1 sweep ~0.5 s
    linearDrag: float = 0.05
    angularDrag: float = 0.3
    fuelBurnRate: float = 0.08
    dt: float = 0.05
    maxSteps: int = 600
    maxLandingSpeed: float = 2.0
    maxLandingTilt: float = 0.15
    maxLandingOmega: float = 0.5
    # Landing-gear geometry (hashed; geometry == physics). legSpan is the
    # horizontal half-distance from the booster base to a leg toe (stance
    # half-width); bodyHalfLen is the base-to-CoM distance. With legDrop (below)
    # they set BOTH the toe positions (src/env/physics.py legToes) and the
    # balance threshold used by episode.py to classify a settled booster as upright:
    # atan2(legSpan, bodyHalfLen + legDrop) ~= 0.32 rad (~18 deg). The actual
    # topple dynamics now emerge from Pymunk rigid-body collision — the episode
    # observer waits for the booster to reach REST_SPEED / REST_OMEGA, then checks
    # the settled theta against this threshold.
    # NOTE: settleTime is DORMANT — it previously fed a one-step tip-over prediction
    # (tipOverAtTouchdown, removed). Pymunk collision now settles the booster
    # physically; settleTime is not read by anything. It is kept hashed only to avoid
    # bumping the world hash without a deliberate world edit.
    legSpan: float = 0.9
    bodyHalfLen: float = 1.8
    settleTime: float = 0.4    # DORMANT — no longer read (see note above)
    # @CONFIG[world.legDrop]: how far a leg toe sits BELOW the booster base when
    # upright (meters), along the body-down axis. With legSpan it places the two
    # toes (src/env/physics.py legToes + the Pymunk leg shapes); the LOWEST toe —
    # not the base — is what collides with the ground. Hashed (geometry == physics)
    # and shared by the renderer so the drawn toes ARE the collidable toes.
    legDrop: float = 0.9
    # @CONFIG[world.settleStepCap]: DORMANT. Under the old scripted-pivot settling
    # model this capped the pivot iterations; the Pymunk solver now does settling
    # for real and the only episode-length bound is world.maxSteps. Still hashed
    # (kept to avoid another world re-hash; some tests use it as a loop budget).
    # Remove it in a future deliberate world edit if desired.
    settleStepCap: int = 120


@dataclass(frozen=True)
class RewardConfig:
    preset: str = 'baseline'
    terminalSuccess: float = 1.0
    terminalCrash: float = -1.0
    gentlenessBonus: float = 0.5
    centeringBonus: float = 0.5
    shapingCoef: float = 1.0
    shapingAnneal: str = 'linear'    # linear | none
    controlCost: float = 0.01


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 3.0e-4
    gamma: float = 0.99
    gaeLambda: float = 0.95
    clipEps: float = 0.2
    epochs: int = 10
    minibatchSize: int = 64
    rolloutSteps: int = 2048
    entCoef: float = 0.0
    vfCoef: float = 0.5
    maxGradNorm: float = 0.5
    numEnvs: int = 16
    evalSeeds: tuple[int, ...] = (0, 1, 2)
    evalEpisodes: int = 40
    evalEvery: int = 5
    totalIters: int = 300
    hidden: tuple[int, ...] = (64, 64)
    device: str = 'auto'    # auto (cuda if present else cpu) | cpu (force fallback)
    # @CONFIG[training.seedWorkers]: how many evalSeeds train concurrently (one
    # OS process per seed; see src/train/parallel.py). 'auto' caps at
    # min(len(evalSeeds), cpu_count); an int pins it; 1 == sequential. Like
    # device, it is a TRAINING concern only and never enters computeWorldHash.
    seedWorkers: int | str = 'auto'    # auto (min(seeds, cpu_count)) | int >= 1


@dataclass(frozen=True)
class CurriculumStage:
    # Spawn randomization ranges, each an inclusive (lo, hi) pair sampled
    # uniformly at episode reset. vy is negative = falling.
    name: str = 'full'
    altitude: tuple[float, float] = (40.0, 52.0)
    xOffset: tuple[float, float] = (-14.0, 14.0)
    vx: tuple[float, float] = (-5.0, 5.0)
    vy: tuple[float, float] = (-12.0, -4.0)
    tilt: tuple[float, float] = (-0.4, 0.4)
    omega: tuple[float, float] = (-0.3, 0.3)


@dataclass(frozen=True)
class CurriculumConfig:
    promoteAt: float = 0.8
    stages: tuple[CurriculumStage, ...] = (CurriculumStage(),)


@dataclass(frozen=True)
class RuntimeConfig:
    watchModel: str = 'best'     # checkpoint WITHIN a model dir: best | seed<N>
    evaluateEpisodes: int = 100


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    mode: str = 'train'              # train | watch | play

    def computeWorldHash(self) -> str:
        """16-hex digest over the world (physics/obs) fields AND the physics-model
        version. A model is loadable iff its stored world hash matches the current
        config's. Reward/training/curriculum knobs are excluded by construction —
        they change how a net is trained, not the world it acts in, so they must
        never block watch/play.

        PHYSICS_MODEL_VERSION is folded in so a change to the SIMULATION MODEL
        itself (not just its tunable fields) invalidates old checkpoints. The
        world config fields are identical before and after the Pymunk rewrite, but
        the dynamics are not — without the version tag a checkpoint trained on the
        old hand-written integrator would load silently against incompatible
        physics. Bump PHYSICS_MODEL_VERSION on any such model change."""
        worldDict = asdict(self.world)
        worldDict['__physicsModel__'] = PHYSICS_MODEL_VERSION
        blob = json.dumps(worldDict, sort_keys=True).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()[:16]


# <agent_context>
#   [ARCH]: Pure validation gate — reads only from cfg fields, raises ValueError
#           with a targeted message on the first bad value found, then returns.
#   [GOTCHA]: Called by loadConfig() immediately before `return cfg`. Any new
#             field added to a dataclass that has a constrained domain MUST also
#             get a matching check here; otherwise bad values silently reach the
#             physics engine.
# </agent_context>
# <agent_guardrail>
#   [CRITICAL]: Validate ONLY fields whose invalid value would produce silent
#               garbage downstream (YAGNI) — no blanket range checks.
#   [CRITICAL]: Keep this a pure function: no prints, no logging, no side-effects.
#               It either raises ValueError or returns None.
#   [VALIDATION]: python -m pytest tests/test_config_loader.py -v must pass.
# </agent_guardrail>
def validateConfig(cfg: Config) -> None:
    """Fail fast on configs that would silently produce garbage downstream.
    Raises ValueError with a specific message on the first problem found."""
    world = cfg.world
    if world.width <= 0 or world.ceiling <= 0:
        raise ValueError('world.width and world.ceiling must be > 0')
    if world.padWidth <= 0 or world.padWidth > world.width:
        raise ValueError('world.padWidth must be in (0, world.width]')
    if world.dt <= 0:
        raise ValueError('world.dt must be > 0')
    if world.maxSteps <= 0:
        raise ValueError('world.maxSteps must be > 0')
    fullMass = world.dryMass + world.fuelMass
    if world.dryMass <= 0 or world.fuelMass <= 0:
        raise ValueError('world.dryMass and world.fuelMass must be > 0')
    if world.maxThrustForce / fullMass <= world.gravity:
        raise ValueError(
            'world.maxThrustForce / (dryMass + fuelMass) must exceed gravity — '
            'a booster that cannot lift its full mass can never arrest its fall',
        )
    if world.throttleResponse <= 0:
        raise ValueError('world.throttleResponse must be > 0')
    if world.gimbalResponse <= 0:
        raise ValueError('world.gimbalResponse must be > 0')
    if world.gimbalArm <= 0 or world.momentInertiaCoef <= 0:
        raise ValueError('world.gimbalArm and world.momentInertiaCoef must be > 0')
    if world.maxGimbal <= 0:
        raise ValueError('world.maxGimbal must be > 0')
    if world.fuelBurnRate <= 0:
        raise ValueError('world.fuelBurnRate must be > 0')
    if world.maxLandingSpeed <= 0 or world.maxLandingTilt <= 0 or world.maxLandingOmega <= 0:
        raise ValueError('landing limits must be > 0')
    if world.legSpan <= 0 or world.bodyHalfLen <= 0:
        raise ValueError('world.legSpan and world.bodyHalfLen must be > 0')
    if world.settleTime < 0:
        raise ValueError('world.settleTime must be >= 0')
    if world.legDrop <= 0:
        raise ValueError('world.legDrop must be > 0')
    if not isinstance(world.settleStepCap, int) or isinstance(world.settleStepCap, bool) or world.settleStepCap <= 0:
        raise ValueError('world.settleStepCap must be an int > 0')

    if cfg.training.device not in ('auto', 'cpu'):
        raise ValueError(
            f'training.device must be auto|cpu, got {cfg.training.device!r}',
        )

    # @INVARIANT: seedWorkers is the literal 'auto' or an int >= 1. A bool is an
    # int subclass in Python, so reject it explicitly (True would read as 1).
    seedWorkers = cfg.training.seedWorkers
    isValidWorkers = seedWorkers == 'auto' or (
        isinstance(seedWorkers, int) and not isinstance(seedWorkers, bool) and seedWorkers >= 1
    )
    if not isValidWorkers:
        raise ValueError(
            f"training.seedWorkers must be 'auto' or an int >= 1, got {seedWorkers!r}",
        )

    if cfg.mode not in ('train', 'watch', 'play'):
        raise ValueError(f'mode must be train|watch|play, got {cfg.mode!r}')
    if cfg.reward.shapingAnneal not in ('linear', 'none'):
        raise ValueError(
            f'reward.shapingAnneal must be linear|none, got {cfg.reward.shapingAnneal!r}',
        )

    curriculum = cfg.curriculum
    if not 0 < curriculum.promoteAt <= 1:
        raise ValueError(f'curriculum.promoteAt must be in (0, 1], got {curriculum.promoteAt!r}')
    if len(curriculum.stages) == 0:
        raise ValueError('curriculum.stages must contain at least one stage')
    for stage in curriculum.stages:
        for key in _STAGE_RANGE_FIELDS:
            lo, hi = getattr(stage, key)
            if lo > hi:
                raise ValueError(
                    f'curriculum stage {stage.name!r}: {key} range is inverted ({lo} > {hi})',
                )
        if stage.altitude[0] <= 0 or stage.altitude[1] >= world.ceiling:
            raise ValueError(
                f'curriculum stage {stage.name!r}: altitude must lie in (0, world.ceiling)',
            )


_STAGE_RANGE_FIELDS = ('altitude', 'xOffset', 'vx', 'vy', 'tilt', 'omega')


def _buildStage(raw: dict) -> CurriculumStage:
    """Convert one YAML stage mapping into a frozen CurriculumStage,
    coercing every [lo, hi] list into a tuple."""
    stageRaw = dict(raw)
    for key in _STAGE_RANGE_FIELDS:
        if key in stageRaw:
            stageRaw[key] = tuple(stageRaw[key])
    return CurriculumStage(**stageRaw)


def loadConfig(path='config.yaml'):
    """Read a YAML control panel into a typed Config."""
    with open(path, 'r', encoding='utf-8') as handle:
        raw = yaml.safe_load(handle) or {}

    trainingRaw = dict(raw.get('training') or {})
    if 'evalSeeds' in trainingRaw:
        trainingRaw['evalSeeds'] = tuple(trainingRaw['evalSeeds'])
    if 'hidden' in trainingRaw:
        trainingRaw['hidden'] = tuple(trainingRaw['hidden'])

    curriculumRaw = dict(raw.get('curriculum') or {})
    if 'stages' in curriculumRaw:
        curriculumRaw['stages'] = tuple(
            _buildStage(stage) for stage in curriculumRaw['stages']
        )

    cfg = Config(
        world=WorldConfig(**(raw.get('world') or {})),
        reward=RewardConfig(**(raw.get('reward') or {})),
        training=TrainingConfig(**trainingRaw),
        curriculum=CurriculumConfig(**curriculumRaw),
        runtime=RuntimeConfig(**(raw.get('runtime') or {})),
        mode=raw.get('mode') or 'train',
    )
    validateConfig(cfg)
    return cfg
