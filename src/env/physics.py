# src/env/physics.py
# <agent_context>
#   [ARCH]: Pymunk rigid-body integrator for the gimbaled booster. BoosterSim is
#           the stateful per-episode simulator: it owns a pymunk.Space with the
#           booster body + leg shapes and a static ground/wall box. Engine, spool,
#           fuel, and transition bookkeeping live here; Pymunk owns geometry + collision.
#           The episode (LandingEnv) owns ONE persistent BoosterSim; it drives
#           BOTH flight AND ground contact through the same unified Pymunk solver —
#           there is no separate settling integrator.
#   [ARCH]: stepPhysics(state, action, world) is a PURE-FUNCTION shim that builds a
#           transient BoosterSim, steps once, and returns a new BoosterState. It is
#           kept for tests that exercise engine/spool/fuel/gimbal logic without a
#           persistent sim. It does NOT implement separate ground-floor clamps or
#           settling; those emerge from Pymunk collision shapes.
#   [GOTCHA]: Coordinate mapping — TWO sign relationships:
#             (1) theta <-> pymunk angle: repo theta>0 leans toward +x; Pymunk
#                 angle is CCW-positive. The body's local +y is body-up. At Pymunk
#                 angle A, local +y in world = (-sin(A), cos(A)). We want body-up =
#                 (sin(theta), cos(theta)), so A = -theta. In code: b.angle = -theta.
#             (2) omega <-> Pymunk angular_velocity: Pymunk angular_velocity is CCW-
#                 positive; our omega is CW-positive (theta growing = leaning more +x).
#                 So omega_repo = -b.angular_velocity.
#   [GOTCHA]: Body origin in Pymunk is the centre of mass. We keep the repo
#             convention that state.x/y is the BASE (not the CoM). Conversion:
#               setState — b.position = base + body_up_world * bodyHalfLen
#                          where body_up_world = (-sin(b.angle), cos(b.angle))
#                          = (sin(theta), cos(theta))
#               getState — base = b.position - body_up_world * bodyHalfLen
#             Both directions expressed purely in terms of b.angle to avoid storing
#             theta redundantly.
#   [GOTCHA]: Force/torque split for faithful dynamics. Thrust is applied as
#             (a) a force at the CoM (gives the correct linear accel = F/mass) PLUS
#             (b) an explicit torque = +(thrustForce * gimbalArm * maxGimbal) * gimbal
#             (Pymunk CCW-positive). This exactly reproduces the old formula
#             alpha = -(thrustForce * gimbalArm * maxGimbal / I) * gimbal because
#             d(omega_repo)/dt = -torque_ccw/I = -(thrustForce*gimbalArm*maxGimbal*gimbal)/I.
#             Using gimbalArm (config, 1.0) rather than the geometric bodyHalfLen (1.8)
#             preserves the original rotational authority exactly.
#   [GOTCHA]: Pymunk body sleeping is DISABLED (sleep_time_threshold=inf). Without
#             this, a landed booster would sleep and stop responding to forces (e.g.
#             a still-firing engine would produce no motion). Never re-enable sleeping.
#   [GOTCHA]: Drag is applied as explicit forces/torques each step, not via
#             space.damping, so the coefficients match the original physics exactly.
#   [GOTCHA]: Pymunk uses explicit Euler (velocity updated first, then position from
#             the OLD velocity in the same step). Consequence: after ONE free-fall
#             step from rest, vy becomes -g*dt but y stays at its initial value; y
#             only starts decreasing from step 2 onward. Tests must assert over
#             multiple steps, not exact single-step position.
#   [GOTCHA]: No continuous collision detection (CCD). The static segments confine
#             the booster only at realistic in-episode speeds (<= ~20 m/s). At
#             extreme velocities (vx/vy >> 20 m/s) the body can penetrate a segment
#             by up to v*dt per step before the contact solver pushes it back. Do not
#             test containment with pathological velocities.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Keep engine/spool/fuel logic IDENTICAL to the original stepPhysics.
#               Only the INTEGRATION of motion changes (Pymunk vs manual Euler).
#               Any edit to the spool, suicide-burn engine,
#               or fuel-burn blocks must be regression-tested against test_physics.py.
#   [CRITICAL]: theta = -b.angle; omega_repo = -b.angular_velocity; base = CoM - body_up*bhl.
#               The @TAG[angle-map] comment marks every conversion site. Do not add
#               conversions elsewhere without adding the tag.
#   [CRITICAL]: Never set space.damping (it overrides our manual drag). Never enable
#               body sleeping. Never apply gravity as a force (Pymunk does it via
#               space.gravity automatically).
#   [CRITICAL]: stepPhysics() is a pure-function shim. Do NOT add rng calls,
#               episode/termination logic, or reward math here.
#   [VALIDATION]: python -m pytest tests/test_physics.py -v
# </agent_guardrail>
"""Pymunk rigid-body simulator for the 2D booster.

Public surfaces:
  BoosterSim  — persistent space owned by the env; call setState()/step()/getState().
  stepPhysics — pure-function shim (builds a transient sim) for tests of engine/spool/fuel.
  legToes     — toe-geometry helper; the single source used by episode.py contact
                detection AND render.py, so the drawn toes are the collidable toes.
  boosterCoM  — centre-of-mass helper (test-only at present).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

SUICIDE_ON_THRESHOLD = 0.5   # env-action throttle above this = engine ON (full)

# A tiny non-zero radius makes Pymunk's GJK solver robust for the thin leg
# segments colliding with the ground segment. (The hull is a box Poly — no radius.)
_LEG_RADIUS = 0.04

# Body half-width (meters). MUST match src/runtime/render.BODY_HALF_W so the drawn
# hull equals the collidable hull. (Half-LENGTH is world.bodyHalfLen, a hashed
# field both physics and render read — only the width is a shared literal.)
BODY_HALF_W = 0.35

# Contact material params for the static geometry and booster shapes. Inelastic
# (no bounce) everywhere; high grip on the surfaces the legs land on, low on the
# walls/ceiling (the booster should slide along, not stick to, the box edges).
_ELASTICITY = 0.0          # inelastic — booster does not bounce
_GROUND_FRICTION = 0.9     # ground + legs: high grip so a planted leg holds
_HULL_FRICTION = 0.5       # hull-on-ground (a toppled booster) — moderate grip
_WALL_FRICTION = 0.0       # frictionless side walls / ceiling
_SEGMENT_THICKNESS = 0.1   # static-segment half-thickness (just needs > 0)

# @TAG[substeps]: Pymunk has NO continuous collision detection, so a fast-moving
# toe can sink deep into the ground in a single dt before the contact solver
# reacts — then get shoved back out over several frames. That deep-penetration +
# ooze-back-out is the visible "rubbery / not-rigid" feel on hard crashes. We
# fix it by advancing the solver in _SUBSTEPS smaller sub-ticks of dt/_SUBSTEPS
# per env step: a toe travels at most dt/_SUBSTEPS worth before contact is seen,
# so penetration stays tiny and the booster behaves rigidly. Measured: a 15 m/s
# slam goes from ~0.4-0.76 m of toe penetration at 1 substep to ~0.0 m at 4.
# Forces are RE-APPLIED each sub-tick (Pymunk zeroes body.force/torque after every
# space.step), and drag is recomputed per sub-tick from the current velocity.
# _SOLVER_ITERATIONS / _COLLISION_SLOP stiffen contact resolution.
_SUBSTEPS = 4
_SOLVER_ITERATIONS = 20    # default is 10; more = stiffer, less spongy contacts
_COLLISION_SLOP = 0.01     # default 0.1 m allowed overlap; tighter = crisper rest


# ---------------------------------------------------------------------------
# @ANCHOR[leg-geometry]: world-frame toe positions for a given BoosterState.
# Body-up unit = (sin theta, cos theta); body-right = (cos theta, -sin theta).
# Toe at half-span legSpan (side +1 = +x toe, -1 = -x toe) and legDrop BELOW the
# base along the body-down axis. SINGLE source of toe geometry used by the
# episode contact detection and the renderer.
# ---------------------------------------------------------------------------
def legToes(state, world):
    """Return ((plusX, plusY), (minusX, minusY)) world-frame leg-toe positions."""
    upX, upY = math.sin(state.theta), math.cos(state.theta)
    rightX, rightY = math.cos(state.theta), -math.sin(state.theta)
    toes = []
    for side in (1, -1):
        toeX = state.x + side * rightX * world.legSpan - upX * world.legDrop
        toeY = state.y + side * rightY * world.legSpan - upY * world.legDrop
        toes.append((toeX, toeY))
    return toes[0], toes[1]


# @ANCHOR[com]: world-frame centre of mass — bodyHalfLen up the body axis from base.
def boosterCoM(state, world):
    """Return (comX, comY) world-frame centre of mass."""
    return (
        state.x + math.sin(state.theta) * world.bodyHalfLen,
        state.y + math.cos(state.theta) * world.bodyHalfLen,
    )


# @ANCHOR[booster-state]: full mutable state of the booster.
@dataclass
class BoosterState:
    x: float = 0.0       # horizontal offset from pad center, m
    y: float = 0.0       # altitude of booster base above ground, m
    vx: float = 0.0      # m/s
    vy: float = 0.0      # m/s (negative = falling)
    theta: float = 0.0   # body tilt from vertical, rad; positive = toward +x
    omega: float = 0.0   # rad/s, CW-positive (theta-dot convention)
    fuel: float = 1.0    # tank fraction remaining in [0, 1]
    spool: float = 0.0   # engine ACTUAL throttle in [0, 1], lags the command
    engineTransitions: int = 0   # on/off state-changes used (suicideBurn: max 2)
    engineCommandedOn: bool = False   # suicideBurn: latched intent (survives spool decay)


# ---------------------------------------------------------------------------
# @ANCHOR[booster-sim]: persistent Pymunk space owned by LandingEnv.
# ---------------------------------------------------------------------------
class BoosterSim:
    """Persistent Pymunk rigid-body simulator for one episode.

    Lifecycle:
        sim = BoosterSim(world)
        obs_state = sim.setState(initial_state)
        for each step:
            new_state = sim.step(action, world)
        # reset for next episode:
        sim.setState(new_initial_state)

    Coordinate conventions (see header block):
        b.angle = -theta  (Pymunk CCW-pos vs repo CW-pos theta)
        omega_repo = -b.angular_velocity
        b.position = CoM world = base + body_up * bodyHalfLen
    """

    def __init__(self, world):
        """Build the Pymunk space. Static ground/wall segments are permanent;
        the dynamic booster body is added once and repositioned each reset via
        setState(). Call setState() before the first step()."""
        self._world = world
        self._space = pymunk.Space()
        # @CONFIG[world.gravity]: applied by Pymunk each step automatically.
        self._space.gravity = (0.0, -world.gravity)
        # @RISK: sleeping would freeze a landed booster's response to thrust.
        self._space.sleep_time_threshold = float('inf')   # disable sleeping
        # @TAG[substeps]: stiffer contact resolution so hard impacts resolve
        # rigidly rather than spongily (paired with sub-stepping in step()).
        self._space.iterations = _SOLVER_ITERATIONS
        self._space.collision_slop = _COLLISION_SLOP

        self._buildStaticGeometry(world)
        self._buildDynamicBody(world)

        # Fuel/spool/engine bookkeeping — Pymunk has no concept of these.
        self._fuel: float = 1.0
        self._spool: float = 0.0
        self._engineTransitions: int = 0
        self._engineCommandedOn: bool = False

    # @TAG[static-geometry]: ground + side walls + ceiling as static segments.
    def _buildStaticGeometry(self, world):
        """Add inelastic ground, side walls, and ceiling to the static body.
        These confine the booster identically to the old explicit clamps."""
        sb = self._space.static_body
        hw = world.width / 2.0
        ceil = world.ceiling

        # Ground — the legs collide here (high grip so a planted leg holds).
        ground = pymunk.Segment(sb, (-hw, 0.0), (hw, 0.0), _SEGMENT_THICKNESS)
        ground.friction = _GROUND_FRICTION
        ground.elasticity = _ELASTICITY
        self._space.add(ground)

        # Side walls — inelastic, frictionless (booster slides along, not sticks).
        for sx in (-hw, hw):
            wall = pymunk.Segment(sb, (sx, 0.0), (sx, ceil + 2.0), _SEGMENT_THICKNESS)
            wall.friction = _WALL_FRICTION
            wall.elasticity = _ELASTICITY
            self._space.add(wall)

        # Ceiling — prevents escape upward.
        ceiling = pymunk.Segment(sb, (-hw, ceil), (hw, ceil), _SEGMENT_THICKNESS)
        ceiling.friction = _WALL_FRICTION
        ceiling.elasticity = _ELASTICITY
        self._space.add(ceiling)

    # @TAG[dynamic-body]: the booster body + hull poly + leg segments.
    def _buildDynamicBody(self, world):
        """Create the booster Body with its hull Poly and two leg Segments.

        Body origin = CoM (Pymunk convention). State.x/y is the BASE, which is
        bodyHalfLen BELOW the CoM along the body-up axis. Leg toes in local body
        coordinates are computed from the base local origin (0, -bodyHalfLen):

            plusToe_local  = ( legSpan, -bodyHalfLen - legDrop)
            minusToe_local = (-legSpan, -bodyHalfLen - legDrop)

        Mass/moment are set to the FULL-TANK values here and updated each step
        via _updateMass(). Shapes carry no mass (density=0 is the default when
        body mass is set explicitly).
        """
        bhl = world.bodyHalfLen
        bhw = BODY_HALF_W   # shared with render.BODY_HALF_W (module constant)
        fullMass = world.dryMass + world.fuelMass * 1.0   # full tank
        # @CONFIG[world.momentInertiaCoef]: exact scalar from config so rotational
        # authority stays identical to the original physics.
        moment = world.momentInertiaCoef * fullMass
        self._body = pymunk.Body(fullMass, moment)
        # position / angle will be set by setState(); add the body now.
        self._space.add(self._body)

        # Hull polygon — a box centered at the body's CoM (origin in local frame).
        # Upward from -bodyHalfLen to +bodyHalfLen in local y, width 2*bodyHalfW.
        hull = pymunk.Poly.create_box(self._body, (2 * bhw, 2 * bhl))
        hull.friction = _HULL_FRICTION
        hull.elasticity = _ELASTICITY
        # Shapes with no density/mass don't affect body mass when explicit mass is set.
        self._space.add(hull)

        # Leg segments: local endpoints relative to the body origin (CoM).
        # Base in local = (0, -bodyHalfLen). Toes at (+/-legSpan, -bodyHalfLen - legDrop).
        baseLoc = (0.0, -bhl)
        for sx in (1, -1):
            toeLoc = (sx * world.legSpan, -bhl - world.legDrop)
            leg = pymunk.Segment(self._body, baseLoc, toeLoc, _LEG_RADIUS)
            leg.friction = _GROUND_FRICTION
            leg.elasticity = _ELASTICITY
            self._space.add(leg)

    # @TAG[mass-update]: keep body mass/moment in sync with current fuel.
    def _updateMass(self, fuel: float):
        """Recompute and set body mass and moment from current fuel fraction.
        Called at the start of each step() before force application."""
        mass = self._world.dryMass + self._world.fuelMass * fuel
        self._body.mass = mass
        self._body.moment = self._world.momentInertiaCoef * mass

    # @TAG[angle-map]: canonical angle/omega conversion helpers.
    # pymunk_angle = -theta; omega_repo = -angular_velocity.
    def _thetaFromBody(self) -> float:
        return -self._body.angle

    def _comToBase(self) -> tuple[float, float]:
        """Extract base (x, y) from the current body position and angle."""
        bhl = self._world.bodyHalfLen
        ang = self._body.angle
        # body-up in world = (-sin(ang), cos(ang))
        baseX = self._body.position.x - (-math.sin(ang)) * bhl
        baseY = self._body.position.y - math.cos(ang) * bhl
        return baseX, baseY

    def _baseToCoM(self, baseX: float, baseY: float, angle: float) -> tuple[float, float]:
        """Compute CoM world position from base and Pymunk angle."""
        bhl = self._world.bodyHalfLen
        # body-up in world = (-sin(ang), cos(ang))
        comX = baseX + (-math.sin(angle)) * bhl
        comY = baseY + math.cos(angle) * bhl
        return comX, comY

    # @SIDEFX: repositions the Pymunk body; resets all engine bookkeeping.
    def setState(self, state: BoosterState) -> None:
        """Place the booster at the given BoosterState. Must be called after
        __init__ and before each episode. Resets all Pymunk body kinematics."""
        angle = -state.theta    # @TAG[angle-map]: repo theta -> pymunk angle
        comX, comY = self._baseToCoM(state.x, state.y, angle)
        self._body.position = (comX, comY)
        self._body.angle = angle
        self._body.velocity = (state.vx, state.vy)
        self._body.angular_velocity = -state.omega   # @TAG[angle-map]
        self._body.force = (0.0, 0.0)
        self._body.torque = 0.0

        self._fuel = state.fuel
        self._spool = state.spool
        self._engineTransitions = state.engineTransitions
        self._engineCommandedOn = state.engineCommandedOn

        self._updateMass(self._fuel)

    def getState(self) -> BoosterState:
        """Read the current Pymunk body kinematics back into a BoosterState."""
        baseX, baseY = self._comToBase()
        theta = self._thetaFromBody()    # @TAG[angle-map]
        omega = -self._body.angular_velocity   # @TAG[angle-map]
        return BoosterState(
            x=baseX, y=baseY,
            vx=self._body.velocity.x, vy=self._body.velocity.y,
            theta=theta, omega=omega,
            fuel=self._fuel, spool=self._spool,
            engineTransitions=self._engineTransitions,
            engineCommandedOn=self._engineCommandedOn,
        )

    # @ANCHOR[sim-step]: the main per-timestep integrator.
    def step(self, action, world) -> BoosterState:
        """Advance one physics timestep.

        Engine/spool/fuel logic is IDENTICAL to the original stepPhysics.
        Motion integration is delegated to Pymunk (space.step).

        Returns a new BoosterState read from the Pymunk body after the step.
        `world` may differ from the construction world if hot-swapped (uncommon).
        """
        rawThrottle = min(max(float(action[0]), 0.0), 1.0)
        gimbal = min(max(float(action[1]), -1.0), 1.0)

        hasFuel = self._fuel > 0.0

        # @TAG[engine-logic]: binary suicide-burn engine — the ONLY engine mode.
        # Fires at FULL or not at all; at most two state-changes (off->on ignite,
        # on->off cut) then the engine locks. engineCommandedOn is the latched
        # intent so it survives spool decay.
        currentlyOn = self._engineCommandedOn
        desiredOn = rawThrottle > SUICIDE_ON_THRESHOLD
        transitions = self._engineTransitions
        if transitions >= 2:
            engineOn = currentlyOn
        elif desiredOn != currentlyOn and hasFuel:
            engineOn = desiredOn
            transitions += 1
        else:
            engineOn = currentlyOn
        engineCommandedOn = engineOn
        effectiveCmd = 1.0 if (engineOn and hasFuel) else 0.0

        # First-order spool lag.
        maxStep = world.throttleResponse * world.dt
        spool = self._spool + max(-maxStep, min(effectiveCmd - self._spool, maxStep))
        spool = min(max(spool, 0.0), 1.0)

        thrustForce = spool * world.maxThrustForce
        mass = world.dryMass + world.fuelMass * self._fuel
        # Update Pymunk body mass/moment to current fuel level.
        self._updateMass(self._fuel)

        delta = gimbal * world.maxGimbal

        # @TAG[substeps]: advance the solver in _SUBSTEPS sub-ticks of dt/_SUBSTEPS
        # so a fast toe cannot sink deep into the ground in one tick (Pymunk has no
        # CCD). Pymunk clears body.force/torque after every space.step, so thrust,
        # gimbal torque, and drag are RE-APPLIED inside each sub-tick. Thrust
        # direction and drag are recomputed from the CURRENT angle/velocity each
        # sub-tick (more accurate than freezing them across the full dt). Gravity
        # is applied automatically by Pymunk on every sub-step. Gimbal torque and
        # spool are held fixed across the sub-ticks (they are per-command, not
        # per-sub-tick, quantities).
        subDt = world.dt / _SUBSTEPS
        for _ in range(_SUBSTEPS):
            theta = self._thetaFromBody()    # @TAG[angle-map]
            phi = theta + delta              # thrust direction from vertical
            # @TAG[force-torque-split]: thrust as force at CoM (no spurious torque)
            # + an explicit gimbal torque, so rotational dynamics match the original
            # alpha = -(thrustForce * gimbalArm * maxGimbal / I) * gimbal. In Pymunk
            # CCW convention that torque is +(thrustForce * gimbalArm * maxGimbal * gimbal).
            Fx = math.sin(phi) * thrustForce
            Fy = math.cos(phi) * thrustForce
            self._body.apply_force_at_world_point((Fx, Fy), tuple(self._body.position))

            # @TAG[angular-drag]: torque_ccw = +angularDrag * angular_velocity * I
            # (CW-convention angularDrag*omega drag). @TAG[linear-drag]: -linearDrag*m*v
            # at the CoM. Both recomputed from the live body state each sub-tick.
            inertia = self._body.moment
            gimbalTorque = (thrustForce * world.maxGimbal * world.gimbalArm) * gimbal
            dragTorque = world.angularDrag * self._body.angular_velocity * inertia
            self._body.torque = gimbalTorque + dragTorque
            self._body.apply_force_at_world_point(
                (
                    -world.linearDrag * mass * self._body.velocity.x,
                    -world.linearDrag * mass * self._body.velocity.y,
                ),
                tuple(self._body.position),
            )
            self._space.step(subDt)

        # Fuel burn proportional to spool (post-step spool value).
        fuel = self._fuel
        if hasFuel:
            fuel = max(0.0, fuel - spool * world.fuelBurnRate * world.dt)

        # Commit bookkeeping.
        self._fuel = fuel
        self._spool = spool
        self._engineTransitions = transitions
        self._engineCommandedOn = engineCommandedOn

        return self.getState()


# ---------------------------------------------------------------------------
# @ANCHOR[step-physics-shim]: pure-function compatibility shim.
# Builds a transient BoosterSim, performs one step, returns a new BoosterState.
# Used by tests that exercise engine/spool/fuel/gimbal logic in isolation.
# Confinement (ground, walls, ceiling) is handled by Pymunk inelastic shapes,
# NOT by explicit clamps. Tests must assert the PHYSICAL behavior (see header).
# ---------------------------------------------------------------------------
def stepPhysics(state: BoosterState, action, world) -> BoosterState:
    """Pure-function shim: build a transient BoosterSim, step once, return state.
    Keeps the original API for tests that exercise engine/spool/fuel/gimbal logic
    in isolation without holding a persistent sim. NOT used by runtime code
    (LandingEnv owns a persistent BoosterSim); it is a test-ergonomics helper."""
    sim = BoosterSim(world)
    sim.setState(state)
    return sim.step(action, world)
