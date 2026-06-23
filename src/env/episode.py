# src/env/episode.py
# <agent_context>
#   [ARCH]: Gym-style single-agent landing environment. Composes the persistent
#           Pymunk simulator (BoosterSim in src/env/physics.py), the obs contract
#           (src/env/spaces.py) and the reward module (src/env/rewards.py).
#           Contains neither physics equations nor reward math of its own.
#   [ARCH]: Single-phase episode model — BoosterSim runs flight AND ground contact
#           in one unified solver. The episode OBSERVES the physical state each
#           step and classifies the outcome: touchdown (contact detected via legToes)
#           triggers REST detection; once the booster has settled (low speed + low
#           spin), the outcome is SUCCESS or CRASH based on uprightness, pad
#           position, and impact speed. No scripted pivot or settleVerdict — the
#           Pymunk solver decides where the booster comes to rest.
#   [GOTCHA]: reset(rng) is the ONLY place randomness enters. step() is fully
#             deterministic — do not add rng calls there.
#   [GOTCHA]: self.state is a PROPERTY backed by _state. Assigning to self.state
#             sets an internal dirty flag (_stateDirty) so step() knows to call
#             sim.setState() before the next physics advance. This supports the
#             curriculum/test pattern of hand-placing the booster between steps
#             without requiring callers to invoke sim.setState() explicitly.
#   [GOTCHA]: setStage() takes effect on the NEXT reset by design — switching
#             curriculum stages mid-episode would corrupt the in-flight episode.
#   [GOTCHA]: Contact detection uses the TOE position (legToes from physics.py),
#             not the body base y. Pymunk leg segments (radius 0.04 m) resting on
#             the ground segment (thickness 0.1 m) produce toe geometric y ≈
#             0.08-0.14 m at rest (upright). Use _TOE_CONTACT_EPS = 0.15.
#   [GOTCHA]: Impact speed is captured from PREVSTATE (approach velocity before
#             the solver arrests it). The post-step velocity may be floor-clamped
#             by Pymunk collision. This preserves the FLOOR_CLAMP_EATS_IMPACT_SPEED
#             lesson from the previous scripted-pivot episode model.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Do NOT add reward arithmetic inside step(). computeReward() in
#               rewards.py is the single source of truth for reward logic.
#   [CRITICAL]: terminated (success/crash) and truncated (timeout) are MUTUALLY
#               EXCLUSIVE. Never set both True simultaneously.
#   [CRITICAL]: Outcome classification order: REST detection -> timeout. There is
#               no out-of-bounds outcome — walls confine the world.
#   [CRITICAL]: STAND_TILT is derived from leg geometry (atan2(legSpan,
#               bodyHalfLen + legDrop)). Do not hard-code a number; always
#               recompute from world config so it stays in sync with geometry.
#   [CRITICAL]: _stateDirty is only True after external assignment to self.state.
#               Inside step(), self._state = ... must NOT set the flag, or the
#               sim would be needlessly re-synced every step. Use the raw attribute
#               name (_state) for internal writes.
#   [VALIDATION]: python -m pytest tests/test_episode.py -v
# </agent_guardrail>
"""LandingEnv: drop a booster, fly it to the pad.

Gym-style API:
    reset(rng) -> obs
    step(action) -> obs, reward, terminated, truncated, info

`stage` (a CurriculumStage) defines the spawn randomization; it defaults to the
final (full-difficulty) stage. Randomness lives ONLY in reset(rng); step is
deterministic. The env knows nothing about PyTorch or policies.

Outcome classification (post-rest):
    success  — upright (|theta| < STAND_TILT), on-pad (|x| <= padWidth/2),
               gentle impact (impactSpeed <= maxLandingSpeed).
    crash    — any other settled state (tilted, off-pad, or hard impact).
    timeout  — self.t >= maxSteps before rest detected.
"""
from __future__ import annotations

import math

from src.env.physics import BoosterState, BoosterSim, legToes
from src.env.rewards import computeReward
from src.env.spaces import encodeObs


# @TAG[contact-eps]: vertical tolerance for toe-ground contact detection. A toe
# whose world-y is at or below this threshold is considered in contact with the
# ground. Pymunk leg segments (radius 0.04 m) resting on a ground segment
# (thickness 0.1 m) produce a toe geometric y ≈ 0.08-0.14 m at rest (upright),
# so this must be at least 0.15. The solver handles exact collision; this is
# only for the episode OBSERVER detecting the first impact step.
_TOE_CONTACT_EPS = 0.15   # m — toe at or below this = ground contact detected

# @TAG[rest-thresholds]: settling detection. The booster is at rest when BOTH
# speed AND angular speed are below these limits. Tuned so normal PdPilot
# landings register REST well within maxSteps; loose enough that micro-
# oscillations from Pymunk contact forces don't prevent detection.
REST_SPEED = 0.5    # m/s — linear speed below which the booster is considered settled
REST_OMEGA = 0.3    # rad/s — angular speed below which the booster is considered settled

# @TAG[stand-tilt]: maximum |theta| for a SUCCESS classification. Derived at
# module init from default world geometry; recomputed per call by _standTilt().
# A booster lying on its side has |theta| ~ pi/2, far above this. The formula
# matches the physical topple threshold: atan2(legSpan, bodyHalfLen + legDrop).
_DEFAULT_STAND_TILT = math.atan2(0.9, 1.8 + 0.9)   # ~0.32 rad for default geometry


# @TAG[stand-tilt-fn]: compute the upright-success tilt threshold from world
# geometry. legSpan / (bodyHalfLen + legDrop) is the tangent of the CoM-over-
# planted-toe topple angle. Above this angle a settled booster has fallen over.
def _standTilt(world) -> float:
    return math.atan2(world.legSpan, world.bodyHalfLen + world.legDrop)


# @ANCHOR[landing-env]: the gym-style episode wrapper.
class LandingEnv:
    def __init__(self, cfg, stage=None):
        self.cfg = cfg
        self.stage = stage if stage is not None else cfg.curriculum.stages[-1]
        self.t = 0
        self.shapingScale = 1.0   # anneal factor; the train loop updates this

        # @DEP[→booster-sim]: persistent Pymunk space; outlives episodes.
        self.sim = BoosterSim(cfg.world)

        # @TAG[state-dirty]: backing store for the state property.
        # _stateDirty is set True whenever self.state is assigned from OUTSIDE
        # (e.g. curriculum hand-placement or test fixture). step() syncs the sim
        # before advancing if the flag is set, then clears it.
        self._state = BoosterState()
        self._stateDirty = False

        # @TAG[impact-tracking]: impact speed latched on first ground contact;
        # None until contact is detected, then set once and never overwritten.
        self._impactSpeed = None
        self._hasTouchedDown = False

        # @TAG[cut-gate]: engine command state latched at first toe contact. A true
        # suicide burn must already be cut (engineCommandedOn False) when it touches.
        self._engineOnAtTouchdown = False

    # @TAG[state-property]: property so external assignment marks the sim dirty.
    @property
    def state(self) -> BoosterState:
        return self._state

    @state.setter
    def state(self, value: BoosterState) -> None:
        self._state = value
        self._stateDirty = True   # external write — sim needs re-sync before next step

    def setStage(self, stage):
        """Curriculum hook — the new stage applies from the next reset()."""
        self.stage = stage

    # @SIDEFX: seeds _state, calls sim.setState, and resets episode counters.
    def reset(self, rng):
        """Spawn the booster from the active stage's ranges with a full tank."""
        stage = self.stage
        # Use _state directly to avoid setting _stateDirty — reset always calls
        # sim.setState explicitly below.
        self._state = BoosterState(
            x=rng.uniform(*stage.xOffset),
            y=rng.uniform(*stage.altitude),
            vx=rng.uniform(*stage.vx),
            vy=rng.uniform(*stage.vy),
            theta=rng.uniform(*stage.tilt),
            omega=rng.uniform(*stage.omega),
            fuel=1.0,
        )
        # @SIDEFX: sync the Pymunk body to the new spawn state.
        self.sim.setState(self._state)
        self._stateDirty = False
        self.t = 0
        self._impactSpeed = None
        self._hasTouchedDown = False
        self._engineOnAtTouchdown = False
        return encodeObs(self._state, self.cfg.world)

    def _hasToeContact(self, state) -> bool:
        """True if any leg toe is at or below _TOE_CONTACT_EPS (ground contact)."""
        plus, minus = legToes(state, self.cfg.world)
        return plus[1] <= _TOE_CONTACT_EPS or minus[1] <= _TOE_CONTACT_EPS

    def _isAtRest(self, state) -> bool:
        """True when the booster has settled: low linear AND angular speed."""
        speed = math.hypot(state.vx, state.vy)
        return speed < REST_SPEED and abs(state.omega) < REST_OMEGA

    def _info(self, state, outcome, impactSpeed):
        return {
            'outcome': outcome,
            'impactSpeed': impactSpeed,
            'x': state.x,
            'y': state.y,
            'fuel': state.fuel,
            'engineOnAtTouchdown': self._engineOnAtTouchdown,
            'engineTransitions': state.engineTransitions,
        }

    def step(self, action):
        """Advance one physics timestep via the persistent BoosterSim.

        Contact and toppling are handled by the Pymunk solver. The episode
        observer detects first ground contact (latches impact speed) and then
        waits for the booster to settle (low speed + low spin) before
        classifying the outcome.

        Returns: obs, reward, terminated, truncated, info.
        """
        prevState = self._state
        world = self.cfg.world

        # @TAG[state-dirty]: if external code mutated self.state (e.g. a test
        # fixture placed the booster by assignment), sync the sim before stepping.
        if self._stateDirty:
            self.sim.setState(self._state)
            self._stateDirty = False

        state = self.sim.step(action, world)
        self.t += 1
        # Write directly to _state to avoid re-triggering _stateDirty.
        self._state = state

        # @TAG[impact-tracking]: first toe contact this episode — latch impact
        # speed from the APPROACH velocity (prevState) before the solver arrests it.
        if not self._hasTouchedDown and self._hasToeContact(state):
            self._hasTouchedDown = True
            self._impactSpeed = math.hypot(prevState.vx, prevState.vy)
            # @TAG[cut-gate]: capture the engine command as the booster ENTERED the
            # contact step (prevState), mirroring impactSpeed's approach-velocity read.
            self._engineOnAtTouchdown = prevState.engineCommandedOn

        impactSpeed = self._impactSpeed if self._hasTouchedDown else 0.0

        # @TAG[outcome-classify]: classify outcome ONLY after touchdown AND rest.
        outcome = None
        if self._hasTouchedDown and self._isAtRest(state):
            standTilt = _standTilt(world)
            isUpright = abs(state.theta) < standTilt
            isOnPad = abs(state.x) <= world.padWidth / 2.0
            isGentle = self._impactSpeed <= world.maxLandingSpeed
            isCutOff = not self._engineOnAtTouchdown
            outcome = 'success' if (isUpright and isOnPad and isGentle and isCutOff) else 'crash'

        if outcome is None and self.t >= world.maxSteps:
            outcome = 'timeout'

        terminated = outcome in ('success', 'crash')
        truncated = outcome == 'timeout'

        reward = computeReward(
            self.cfg, prevState, state, action, outcome, impactSpeed,
            shapingScale=self.shapingScale,
        )

        return (
            encodeObs(state, world),
            reward,
            terminated,
            truncated,
            self._info(state, outcome, impactSpeed),
        )
