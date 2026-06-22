# src/runtime/render.py
# <agent_context>
#   [ARCH]: Presentation only. worldToScreen / keysToControls are pure functions;
#           the Renderer class is the only part that touches a real window. The
#           loop depends on the duck-typed interface (pollIntents/draw/tick/close),
#           so a fake renderer stands in for headless tests.
#   [GOTCHA]: World y is up; screen y is down. worldToScreen flips y and centers
#             x (world x in [-width/2, +width/2]).
#   [GOTCHA]: keysToControls returns a throttle RAMP DIRECTION (-1/0/+1), not a
#             throttle value — the human action source in scripts/play.py holds
#             the ramp state. Gimbal is direct: D/RIGHT -> -1 so the NOSE rotates
#             toward +x ("press right = rotate right"; alpha = -thr*gimbal*torque).
# </agent_context>
# <agent_guardrail>
#   [CRITICAL]: Do NOT import LandingEnv, MLPPolicy, or any policy here — render
#               stays domain-free. A BoosterState + action + HUD lines come in as
#               arguments.
#   [CRITICAL]: Tests set SDL_VIDEODRIVER=dummy before constructing a Renderer.
#               Never call pygame.display.set_mode at import time (only in __init__).
#   [VALIDATION]: python -m pytest tests/test_render.py -v
# </agent_guardrail>
"""pygame-ce window + pure presentation helpers for the runtime modes.

The Renderer draws a world snapshot (sky, ground surface line, pad, booster body
with splayed landing legs and a throttle-proportional gimbal-deflected flame,
HUD text band) and polls input into an Intents struct. The drawn leg toes come
from the shared src.env.physics.legToes helper (world.legSpan out, world.legDrop
below the base), so the drawn toes ARE the physical contact points the settling
physics pivots about."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

PX_PER_M = 11
WINDOW_WIDTH = 440         # world.width 40 m * 11 px
WINDOW_HEIGHT = 660        # world.ceiling 60 m * 11 px
HUD_HEIGHT = 70            # text band below the world area
FPS = 20                   # matches world.dt = 0.05 -> real-time default playback

COLOR_BG = (12, 14, 22)
COLOR_GROUND = (60, 50, 45)
COLOR_GROUND_LINE = (120, 105, 90)   # crisp surface line atop the ground band
COLOR_PAD = (70, 160, 90)
COLOR_BODY = (210, 210, 220)
COLOR_NOSE = (240, 120, 90)
COLOR_FLAME = (250, 180, 60)
COLOR_LEG = (150, 155, 170)          # landing-gear struts
COLOR_TEXT = (220, 220, 220)
COLOR_WALL = (45, 50, 65)

# Body half-WIDTH (meters). Must match the physics hull half-width
# (src/env/physics.BODY_HALF_W) so the drawn hull equals the collidable hull.
# The body half-LENGTH is NOT a constant here — _bodyPolygon reads
# world.bodyHalfLen directly so the drawn hull length can never desync from the
# (hashed) physics body length.
BODY_HALF_W = 0.35
# Landing-gear drawing geometry. The legs splay from the base out to a toe
# whose position comes from src.env.physics.legToes (world.legSpan out,
# world.legDrop below — the SAME points the contact/pivot physics uses, so the
# drawn toes ARE the physical toes; @TAG[leg-toes-shared-geometry]). The hinge
# is a cosmetic attachment point a little up the body so the struts angle
# outward like real deployed gear. LEG_THICKNESS is the strut line width in px.
LEG_HINGE = 0.9            # meters up the body the strut attaches (cosmetic)
LEG_THICKNESS = 3          # px

_HELD_KEYS = (
    pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
    pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d,
)


@dataclass
class Intents:
    """One frame of user input, drained from the event queue by pollIntents.
    Edge events (key-down) are bools; throttleDir/gimbal reflect held keys."""
    quit: bool = False
    togglePause: bool = False
    stepOnce: bool = False
    speedUp: bool = False
    speedDown: bool = False
    reset: bool = False
    throttleDir: float = 0.0   # -1 ramp down, +1 ramp up
    gimbal: float = 0.0        # direct gimbal command in {-1, 0, +1}


def worldToScreen(pos, world):
    """Map world (x, y) — x in [-width/2, +width/2], y in [0, ceiling] up — to
    integer pixels with the ground line at the BOTTOM of the world area."""
    sx = WINDOW_WIDTH / world.width
    sy = WINDOW_HEIGHT / world.ceiling
    px = int(round((pos[0] + world.width / 2.0) * sx))
    py = int(round((world.ceiling - pos[1]) * sy))
    return px, py


def keysToControls(pressed):
    """Map held pygame key codes to (throttleDir, gimbal). W/UP vs S/DOWN ramp
    throttle; A/LEFT vs D/RIGHT command gimbal. Opposing keys cancel.
    Sign note: D/RIGHT -> gimbal -1 so the nose rotates toward +x."""
    up = pygame.K_UP in pressed or pygame.K_w in pressed
    down = pygame.K_DOWN in pressed or pygame.K_s in pressed
    right = pygame.K_RIGHT in pressed or pygame.K_d in pressed
    left = pygame.K_LEFT in pressed or pygame.K_a in pressed
    throttleDir = float(up) - float(down)
    gimbal = float(left) - float(right)
    return throttleDir, gimbal


class Renderer:
    """The pygame-ce window. Constructed with a WorldConfig. draw() paints a
    BoosterState + last action + HUD lines; pollIntents() drains input. The
    loop depends only on pollIntents/draw/tick/close."""

    def __init__(self, world):
        self.world = world
        pygame.init()
        pygame.display.set_caption('booster — land on the pad')
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT + HUD_HEIGHT))
        self.font = pygame.font.SysFont('monospace', 13)
        self.clock = pygame.time.Clock()

    def pollIntents(self):
        intents = Intents()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                intents.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    intents.quit = True
                elif event.key == pygame.K_SPACE:
                    intents.togglePause = True
                elif event.key == pygame.K_n:
                    intents.stepOnce = True
                elif event.key in (pygame.K_EQUALS, pygame.K_RIGHTBRACKET):
                    intents.speedUp = True
                elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                    intents.speedDown = True
                elif event.key == pygame.K_r:
                    intents.reset = True
        pressed = pygame.key.get_pressed()
        held = {k for k in _HELD_KEYS if pressed[k]}
        intents.throttleDir, intents.gimbal = keysToControls(held)
        return intents

    def _bodyPolygon(self, state):
        """Four corners of the booster rectangle. The body extends from the BASE
        at (state.x, state.y) along the body-up axis (tilted by theta). The
        half-length is read from world.bodyHalfLen so the drawn hull matches the
        collidable hull (src/env/physics.py create_box(2*BODY_HALF_W, 2*bodyHalfLen))."""
        bhl = self.world.bodyHalfLen
        upX, upY = math.sin(state.theta), math.cos(state.theta)     # body-up unit
        rightX, rightY = math.cos(state.theta), -math.sin(state.theta)
        cx = state.x + upX * bhl                                     # body center
        cy = state.y + upY * bhl
        corners = []
        for du, dr in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            wx = cx + du * upX * bhl + dr * rightX * BODY_HALF_W
            wy = cy + du * upY * bhl + dr * rightY * BODY_HALF_W
            corners.append(worldToScreen((wx, wy), self.world))
        return corners

    def _legSegments(self, state):
        """Two landing-gear struts as (hingeScreen, toeScreen) pixel pairs. The
        TOE comes from src.env.physics.legToes (world.legSpan out, world.legDrop
        below the base) — the SAME points the contact/pivot physics uses, so the
        drawn toes ARE the physical toes. The hinge (a little up the body) is
        render-local cosmetic geometry."""
        # @TAG[leg-toes-shared-geometry]: toe positions sourced from the shared
        # legToes() pure helper — never duplicate inline so they cannot drift.
        from src.env.physics import legToes
        upX, upY = math.sin(state.theta), math.cos(state.theta)
        rightX, rightY = math.cos(state.theta), -math.sin(state.theta)
        plusToe, minusToe = legToes(state, self.world)
        toes = {1: plusToe, -1: minusToe}
        segments = []
        for side in (1, -1):
            hingeX = state.x + upX * LEG_HINGE + side * rightX * BODY_HALF_W
            hingeY = state.y + upY * LEG_HINGE + side * rightY * BODY_HALF_W
            segments.append((
                worldToScreen((hingeX, hingeY), self.world),
                worldToScreen(toes[side], self.world),
            ))
        return segments

    def draw(self, state, action, hudLines):
        """Paint one frame. `state` is a BoosterState; `action` is the last
        [throttle, gimbal] env action (drives the flame); hudLines is a list of
        up to 4 strings."""
        world = self.world
        self.screen.fill(COLOR_BG)

        # Ground band + crisp surface line + pad. The line marks the y=0 plane —
        # the Pymunk static ground segment the legs physically collide with
        # (src/env/physics.py BoosterSim).
        groundY = worldToScreen((0.0, 0.0), world)[1]
        pygame.draw.rect(
            self.screen, COLOR_GROUND,
            (0, groundY, WINDOW_WIDTH, WINDOW_HEIGHT - groundY + 2),
        )
        pygame.draw.line(
            self.screen, COLOR_GROUND_LINE, (0, groundY), (WINDOW_WIDTH, groundY), 2,
        )
        padLeft = worldToScreen((-world.padWidth / 2.0, 0.0), world)[0]
        padRight = worldToScreen((world.padWidth / 2.0, 0.0), world)[0]
        pygame.draw.rect(
            self.screen, COLOR_PAD, (padLeft, groundY - 3, padRight - padLeft, 6),
        )
        pygame.draw.rect(
            self.screen, COLOR_WALL, (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), 2,
        )

        # Flame: from the base along the thrust direction (theta + gimbal
        # deflection), length proportional to throttle. Drawn under the body.
        throttle = max(0.0, min(float(action[0]), 1.0))
        gimbal = max(-1.0, min(float(action[1]), 1.0))
        if throttle > 0.01 and state.fuel > 0.0:
            phi = state.theta + gimbal * world.maxGimbal
            flameLen = 1.0 + 2.6 * throttle
            tipX = state.x - math.sin(phi) * flameLen
            tipY = state.y - math.cos(phi) * flameLen
            rightX, rightY = math.cos(phi), -math.sin(phi)
            base = [
                worldToScreen(
                    (state.x + s * rightX * 0.25, state.y + s * rightY * 0.25),
                    world,
                )
                for s in (1, -1)
            ]
            pygame.draw.polygon(
                self.screen, COLOR_FLAME, base + [worldToScreen((tipX, tipY), world)],
            )

        # Landing legs: drawn under the body so the hull hides the hinge join.
        # Toes come from the shared legToes() helper (see _legSegments), so they
        # ARE the physical contact points the settling phase pivots about
        # (src/env/episode.py @TAG[rest-verdict]).
        for hinge, toe in self._legSegments(state):
            pygame.draw.line(self.screen, COLOR_LEG, hinge, toe, LEG_THICKNESS)

        # Body + nose marker (nose = top end of the body axis, 2*bodyHalfLen up).
        pygame.draw.polygon(self.screen, COLOR_BODY, self._bodyPolygon(state))
        noseLen = 2 * world.bodyHalfLen
        noseX = state.x + math.sin(state.theta) * noseLen
        noseY = state.y + math.cos(state.theta) * noseLen
        pygame.draw.circle(self.screen, COLOR_NOSE, worldToScreen((noseX, noseY), world), 3)

        # HUD band.
        pygame.draw.rect(
            self.screen, (24, 26, 34), (0, WINDOW_HEIGHT, WINDOW_WIDTH, HUD_HEIGHT),
        )
        for i, line in enumerate(hudLines[:4]):
            surface = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(surface, (6, WINDOW_HEIGHT + 4 + i * 16))
        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(max(1, int(fps)))

    def close(self):
        pygame.quit()
