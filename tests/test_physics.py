# tests/test_physics.py
import dataclasses
import math

import pytest

from src.config.loader import loadConfig
from src.env.physics import BoosterState, BoosterSim, stepPhysics, legToes, boosterCoM


@pytest.fixture
def world():
    return loadConfig('config.yaml').world


def _spinUp(world, throttle, steps=40):
    """Run the engine to steady state so spool ~= the commanded throttle."""
    state = BoosterState(y=40.0, fuel=1.0)
    for _ in range(steps):
        state = stepPhysics(state, [throttle, 0.0], world)
    return state


def test_freeFallFromRest(world):
    # @TAG[substeps]: with no thrust the booster free-falls under gravity. Each env
    # step advances the solver in _SUBSTEPS sub-ticks, so after one full dt the
    # speed is ~g*dt (to within an O(dt^2) sub-tick integration difference — Pymunk
    # explicit Euler over sub-ticks does not land exactly on -g*dt) and the position
    # has ALREADY begun to drop within the same step (unlike a single explicit-Euler
    # tick, sub-ticks 2+ move under the velocity gained in earlier sub-ticks).
    state = BoosterState(x=0.0, y=30.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=1.0)
    nxt = stepPhysics(state, [0.0, 0.0], world)
    assert nxt.vy == pytest.approx(-world.gravity * world.dt, rel=0.02)   # ~g*dt
    assert nxt.vy < 0.0             # falling
    assert nxt.y < state.y          # position already dropping (sub-stepped)
    assert abs(nxt.theta) < 1e-9
    assert abs(nxt.omega) < 1e-9
    assert nxt.fuel == 1.0          # zero throttle burns nothing
    assert nxt.spool == 0.0         # engine stays off

    # Over more steps it keeps accelerating downward.
    sim = BoosterSim(world)
    sim.setState(state)
    sim.step([0.0, 0.0], world)
    s2 = sim.step([0.0, 0.0], world)
    assert s2.y < nxt.y             # still falling
    assert s2.vy < nxt.vy           # speed increasing (more negative)
    assert s2.vy < nxt.vy          # still accelerating downward


def test_massScalesAcceleration(world):
    # Same spool, heavier booster (full tank) accelerates less than a light one.
    full = BoosterState(y=40.0, fuel=1.0, spool=1.0)
    light = BoosterState(y=40.0, fuel=0.01, spool=1.0)   # tiny fuel so it still fires
    aFull = stepPhysics(full, [1.0, 0.0], world).vy - full.vy
    aLight = stepPhysics(light, [1.0, 0.0], world).vy - light.vy
    assert aLight > aFull          # lighter -> more upward accel per unit spool


def test_spoolLagsTowardCommand(world):
    state = BoosterState(y=40.0, fuel=1.0, spool=0.0)
    nxt = stepPhysics(state, [1.0, 0.0], world)
    step = world.throttleResponse * world.dt
    assert nxt.spool == pytest.approx(min(step, 1.0))     # moved at most one step
    assert nxt.spool < 1.0                                 # not instant


def test_spoolAsymptotesToCommand(world):
    state = _spinUp(world, 0.8, steps=60)
    assert state.spool == pytest.approx(0.8, abs=1e-3)


def test_minThrottleFloorOnceLit(world):
    # Commanding 0.2 (below minThrottle but above cutoff) spools toward minThrottle.
    state = BoosterState(y=40.0, fuel=1.0, spool=0.0)
    for _ in range(60):
        state = stepPhysics(state, [0.2, 0.0], world)
    assert state.spool == pytest.approx(world.minThrottle, abs=1e-3)


def test_throttleCutoffShutsEngineOff(world):
    lit = _spinUp(world, 1.0, steps=40)
    off = stepPhysics(lit, [0.0, 0.0], world)        # command below cutoff
    assert off.spool < lit.spool                      # spool decaying toward 0


def test_emptyBoosterCanStillHover(world):
    # Design guarantee: a near-empty booster's min-throttle thrust stays below
    # its weight, so it is not forced upward — hovering remains achievable.
    mass = world.dryMass + world.fuelMass * 0.0
    minAccel = world.minThrottle * world.maxThrustForce / mass
    assert minAccel < world.gravity


def test_tiltedThrustGoesSideways(world):
    lit = _spinUp(world, 1.0, steps=40)
    tilted = stepPhysics(dataclasses.replace(lit, theta=0.1), [1.0, 0.0], world)
    assert tilted.vx > tilted.vy * 0     # positive tilt pushes toward +x
    assert tilted.vx > 0


def test_gimbalTorqueSignAndSpoolScaling(world):
    lit = _spinUp(world, 1.0, steps=40)
    turned = stepPhysics(lit, [1.0, 1.0], world)
    assert turned.omega < lit.omega                   # +gimbal rotates nose toward -x
    dead = BoosterState(y=40.0, fuel=0.0, spool=0.0)
    nxt = stepPhysics(dead, [1.0, 1.0], world)
    assert nxt.omega == pytest.approx(0.0)            # no thrust = no steering


def test_fuelBurnsOnSpoolNotCommand(world):
    # Burn is proportional to the (post-step) ACTUAL spool, not the command. From
    # spool 0 the engine only reaches one ramp step, so it burns far less than a
    # fully-spooled engine would at the same command.
    cold = stepPhysics(BoosterState(y=40.0, fuel=1.0, spool=0.0), [1.0, 0.0], world)
    coldBurn = 1.0 - cold.fuel
    step = world.throttleResponse * world.dt
    assert coldBurn == pytest.approx(step * world.fuelBurnRate * world.dt, abs=1e-6)
    # At full spool, fuel burns at the full rate.
    lit = _spinUp(world, 1.0, steps=40)
    after = stepPhysics(lit, [1.0, 0.0], world)
    assert (lit.fuel - after.fuel) == pytest.approx(world.fuelBurnRate * world.dt, abs=1e-4)


def test_deadEngineIsFreeFall(world):
    # Out of fuel -> commanding full throttle produces NO thrust: the booster just
    # free-falls at ~g*dt (within the O(dt^2) sub-tick tolerance, @TAG[substeps]).
    empty = BoosterState(y=30.0, fuel=0.0)
    nxt = stepPhysics(empty, [1.0, 0.0], world)
    assert nxt.vy == pytest.approx(-world.gravity * world.dt, rel=0.02)   # thrust is dead
    assert nxt.vy < 0.0
    assert nxt.fuel == 0.0
    assert nxt.spool == 0.0


def test_deterministicAndPure(world):
    state = BoosterState(x=1.0, y=20.0, vx=-2.0, vy=-5.0, theta=0.2, omega=-0.1, fuel=0.7)
    snapshot = dataclasses.replace(state)
    a = stepPhysics(state, [0.6, -0.3], world)
    b = stepPhysics(state, [0.6, -0.3], world)
    assert a == b
    assert state == snapshot           # input not mutated


def test_actionClippedDefensively(world):
    wild = stepPhysics(BoosterState(y=30.0), [99.0, -99.0], world)
    sane = stepPhysics(BoosterState(y=30.0), [1.0, -1.0], world)
    assert wild == sane


def test_sideWallsClampAndKillInwardVelocity(world):
    # @TAG[pymunk-wall]: Pymunk walls use inelastic collision without CCD. At
    # moderate speeds (<= ~20 m/s, typical for in-episode dynamics) the contact
    # solver reverses the velocity and confines the body within ~vx*dt of the wall.
    # The key invariant: the booster reverses direction and does not escape the box.
    # We use vx=5 (realistic episode range) and assert:
    #   (a) the booster NEVER exceeds width/2 + small_penetration_eps over several steps,
    #   (b) vx reverses sign within a handful of steps (velocity kills inward component).
    halfW = world.width / 2.0
    # Penetration tolerance: body can be at most vx*dt = 5*0.05 = 0.25 m past wall.
    wallEps = 0.3

    sim = BoosterSim(world)
    sim.setState(BoosterState(x=halfW - 0.1, y=30.0, vx=5.0))
    reversed_right = False
    for _ in range(15):
        s = sim.step([0.0, 0.0], world)
        assert s.x <= halfW + wallEps, f'Booster escaped right wall: x={s.x:.3f}'
        if s.vx <= 0.0:
            reversed_right = True
            break
    assert reversed_right, 'Right wall never reversed lateral velocity'

    sim2 = BoosterSim(world)
    sim2.setState(BoosterState(x=-halfW + 0.1, y=30.0, vx=-5.0))
    reversed_left = False
    for _ in range(15):
        s2 = sim2.step([0.0, 0.0], world)
        assert s2.x >= -halfW - wallEps, f'Booster escaped left wall: x={s2.x:.3f}'
        if s2.vx >= 0.0:
            reversed_left = True
            break
    assert reversed_left, 'Left wall never reversed lateral velocity'


def test_ceilingClampsUpwardFlight(world):
    # @TAG[pymunk-ceiling]: Pymunk ceiling is an inelastic static segment without CCD.
    # The collision operates on the HULL, whose top is at base + 2*bodyHalfLen.
    # The ceiling segment is at y=world.ceiling. The hull top reaches the ceiling
    # when base = ceiling - 2*bodyHalfLen. We spawn with base 2m below that safe
    # limit (hull not yet touching) and vy=15 so the hull hits the ceiling within
    # a few steps. Assert: vy reverses and the base stays within a small eps of
    # the safe limit (ceiling - 2*bhl + small_overshoot from vy*dt penetration).
    bhl = world.bodyHalfLen
    safeBase = world.ceiling - 2.0 * bhl   # hull_top = ceiling at this base height
    ceilEps = 1.0   # base may overshoot safeBase by up to vy*dt = 15*0.05 = 0.75 m
    sim = BoosterSim(world)
    sim.setState(BoosterState(x=0.0, y=safeBase - 2.0, vy=15.0, fuel=1.0))
    vy_reversed = False
    for _ in range(10):
        s = sim.step([0.0, 0.0], world)
        assert s.y <= safeBase + ceilEps, f'Booster escaped ceiling: y={s.y:.3f}'
        if s.vy <= 0.0:
            vy_reversed = True
            break
    assert vy_reversed, 'Ceiling never reversed upward velocity'


def test_groundFloorClampsAndKillsDownwardVelocity(world):
    # @TAG[pymunk-ground]: The Pymunk ground is an inelastic static segment; the leg
    # segments collide with it. An upright booster rests with its toes on the ground,
    # so the BASE rests at approximately world.legDrop (not 0). The contact solver
    # kills downward velocity within 1-2 steps. Assert: after a gentle fall, the
    # booster reaches near-zero vy and its base stays above legDrop - small_eps.
    sim = BoosterSim(world)
    sim.setState(BoosterState(x=0.0, y=world.legDrop + 0.1, vy=-3.0, fuel=0.5))
    for _ in range(5):
        s = sim.step([0.0, 0.0], world)
        if abs(s.vy) < 0.1:   # settled
            break
    # Base should be at or above legDrop (toes on ground, not sunk through)
    assert s.y >= world.legDrop - 0.1, f'Booster sank through ground: y={s.y:.3f}'
    assert abs(s.vy) < 0.2, f'Downward velocity not killed: vy={s.vy:.3f}'


def test_groundFloorLeavesUpwardVelocityAlone(world):
    # @TAG[pymunk-ground-launch]: A booster resting on the ground (base at legDrop,
    # toes in contact) with upward initial velocity MAINTAINS that upward motion —
    # the ground only kills the downward (into-surface) component. The booster rises.
    # Use base_y = legDrop so the toes start exactly at the ground.
    sim = BoosterSim(world)
    sim.setState(BoosterState(x=0.0, y=world.legDrop, vy=2.0, fuel=1.0))
    for _ in range(3):
        s = sim.step([0.0, 0.0], world)
    assert s.y > world.legDrop   # booster rose above the ground
    assert s.y >= world.legDrop  # never sank below resting position


def test_boosterStateDefaultsEngineTransitionsZero():
    state = BoosterState()
    assert state.engineTransitions == 0


def test_analogStepPreservesEngineTransitions(world):
    # analog mode (default) never changes the transition counter
    state = BoosterState(y=40.0, fuel=1.0, engineTransitions=0)
    nxt = stepPhysics(state, [1.0, 0.0], world)
    assert nxt.engineTransitions == 0


@pytest.fixture
def burnWorld():
    return dataclasses.replace(loadConfig('config.yaml').world, engineMode='suicideBurn')


def test_suicideBurnIgnitesToFullOnPositiveCommand(burnWorld):
    # engine off; a positive engine command ignites (transition #1) and spools UP
    state = BoosterState(y=40.0, fuel=1.0, spool=0.0, engineTransitions=0)
    nxt = stepPhysics(state, [1.0, 0.0], burnWorld)
    assert nxt.engineTransitions == 1
    assert nxt.spool > 0.0                                  # ignited, spooling up
    assert nxt.spool == pytest.approx(burnWorld.throttleResponse * burnWorld.dt)


def test_suicideBurnPartialCommandStillIgnitesFull(burnWorld):
    # any command above the on-threshold is full throttle; below it is off
    on = stepPhysics(BoosterState(y=40.0, fuel=1.0), [0.6, 0.0], burnWorld)
    off = stepPhysics(BoosterState(y=40.0, fuel=1.0), [0.4, 0.0], burnWorld)
    assert on.engineTransitions == 1 and on.spool > 0.0
    assert off.engineTransitions == 0 and off.spool == 0.0


def test_suicideBurnCutoffIsSecondTransition(burnWorld):
    state = BoosterState(y=40.0, fuel=1.0)
    state = stepPhysics(state, [1.0, 0.0], burnWorld)       # ignite (#1)
    assert state.engineTransitions == 1
    state = stepPhysics(state, [1.0, 0.0], burnWorld)       # stay lit, no new transition
    assert state.engineTransitions == 1
    state = stepPhysics(state, [0.0, 0.0], burnWorld)       # cutoff (#2)
    assert state.engineTransitions == 2


def test_suicideBurnLocksAfterTwoTransitions(burnWorld):
    state = BoosterState(y=40.0, fuel=1.0)
    state = stepPhysics(state, [1.0, 0.0], burnWorld)       # ignite (#1)
    state = stepPhysics(state, [0.0, 0.0], burnWorld)       # cutoff (#2)
    transitionsAtLock = state.engineTransitions
    for _ in range(40):
        state = stepPhysics(state, [1.0, 0.0], burnWorld)   # try to relight repeatedly
    assert state.engineTransitions == transitionsAtLock == 2
    assert state.spool == pytest.approx(0.0, abs=1e-6)      # never relit


def test_suicideBurnCutoffFromFullSpoolActuallyShutsOff(burnWorld):
    # A cutoff issued from a FULLY spooled engine must drive thrust to zero and
    # KEEP it there — the engine must not relight itself during spool decay.
    state = BoosterState(y=200.0, fuel=1.0)
    state = stepPhysics(state, [1.0, 0.0], burnWorld)          # ignite
    for _ in range(40):                                        # spool to full
        state = stepPhysics(state, [1.0, 0.0], burnWorld)
    assert state.spool == pytest.approx(1.0, abs=1e-3)         # fully lit
    state = stepPhysics(state, [0.0, 0.0], burnWorld)          # CUTOFF (transition #2)
    assert state.engineTransitions == 2
    for _ in range(40):                                        # hold cutoff (and even try relight)
        state = stepPhysics(state, [1.0, 0.0], burnWorld)
    assert state.spool == pytest.approx(0.0, abs=1e-6)         # stayed off, never relit


def test_suicideBurnTerminalStateFollowsCommandNotDecayTiming(burnWorld):
    # Edge case: a brief cutoff command spends the cutoff transition and the
    # engine's TERMINAL state follows the agent's command (off), regardless of
    # how far spool had decayed when the next command arrived.
    state = BoosterState(y=200.0, fuel=1.0)
    for _ in range(20):
        state = stepPhysics(state, [1.0, 0.0], burnWorld)      # burn, spool ~1.0
    state = stepPhysics(state, [0.0, 0.0], burnWorld)          # cutoff (#2)
    assert state.engineTransitions == 2
    state = stepPhysics(state, [1.0, 0.0], burnWorld)          # try to relight (locked)
    for _ in range(40):
        state = stepPhysics(state, [1.0, 0.0], burnWorld)
    assert state.spool == pytest.approx(0.0, abs=1e-6)         # locked OFF, follows the cutoff


def test_legToesUprightStraddleBaseAndDropBelow(world):
    # Upright at the origin: toes are +/- legSpan horizontally and legDrop below.
    state = BoosterState(x=0.0, y=0.0, theta=0.0)
    (rxX, rxY), (lxX, lxY) = legToes(state, world)
    assert rxX == pytest.approx(world.legSpan)      # side +1 toe to +x
    assert lxX == pytest.approx(-world.legSpan)     # side -1 toe to -x
    assert rxY == pytest.approx(-world.legDrop)
    assert lxY == pytest.approx(-world.legDrop)


def test_legToesLowestToeIsTheLeaningSide(world):
    # Tilt toward +x: the +x toe (side +1) drops lower than the -x toe.
    state = BoosterState(x=0.0, y=5.0, theta=0.25)
    plus, minus = legToes(state, world)
    assert plus[1] < minus[1]


def test_boosterCoMSitsBodyHalfLenUpTheAxis(world):
    state = BoosterState(x=2.0, y=10.0, theta=0.0)
    comX, comY = boosterCoM(state, world)
    assert comX == pytest.approx(2.0)
    assert comY == pytest.approx(10.0 + world.bodyHalfLen)


# @TAG[ground-drop]: upright booster dropped onto the ground rests stably on its legs.
# The PHYSICAL rest position is base_y ~ legDrop (toes touching ground); vy -> 0.
def test_boosterDropsOntoGroundAndRests(world):
    sim = BoosterSim(world)
    sim.setState(BoosterState(x=0.0, y=3.0, vy=0.0, fuel=1.0))
    for _ in range(30):   # 1.5 s at dt=0.05 — well past time to settle
        s = sim.step([0.0, 0.0], world)
    assert abs(s.y - world.legDrop) < 0.15, f'Resting base_y={s.y:.3f}, expected ~legDrop={world.legDrop}'
    assert abs(s.vy) < 0.1, f'Not at rest: vy={s.vy:.3f}'
