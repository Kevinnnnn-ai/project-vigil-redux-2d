# tests/test_render.py
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')   # before any pygame display call

import pygame
import pytest

from src.config.loader import loadConfig
from src.env.physics import BoosterState
from src.runtime.render import (
    Renderer, worldToScreen, keysToControls, WINDOW_WIDTH, WINDOW_HEIGHT,
)


@pytest.fixture
def world():
    return loadConfig('config.yaml').world


def test_worldToScreenCorners(world):
    # Left wall -> x=0; right wall -> x=W; ground -> bottom of world area;
    # ceiling -> y=0.
    assert worldToScreen((-world.width / 2, 0.0), world) == (0, WINDOW_HEIGHT)
    assert worldToScreen((world.width / 2, 0.0), world) == (WINDOW_WIDTH, WINDOW_HEIGHT)
    assert worldToScreen((0.0, world.ceiling), world) == (WINDOW_WIDTH // 2, 0)


def test_keysToControlsMappings():
    up = {pygame.K_w}
    down = {pygame.K_DOWN}
    right = {pygame.K_d}
    left = {pygame.K_LEFT}
    assert keysToControls(up) == (1.0, 0.0)
    assert keysToControls(down) == (-1.0, 0.0)
    # Press right = rotate the NOSE right: gimbal -1 (alpha > 0 in physics).
    assert keysToControls(right) == (0.0, -1.0)
    assert keysToControls(left) == (0.0, 1.0)
    assert keysToControls(up | down) == (0.0, 0.0)        # opposing keys cancel
    assert keysToControls(set()) == (0.0, 0.0)


def test_rendererSmoke(world):
    renderer = Renderer(world)
    state = BoosterState(x=0.0, y=20.0, theta=0.1, fuel=0.8)
    renderer.draw(state, [0.7, -0.2], ['line one', 'line two'])
    intents = renderer.pollIntents()
    assert intents.quit is False
    renderer.close()


# @TAG[leg-toes-shared-geometry]: renderer must source both leg toes from the
# shared legToes() helper so drawn toes are ALWAYS the physical toes.
def test_legSegmentsUseSharedPhysicsToes(world):
    from src.env.physics import legToes
    r = Renderer(world)
    try:
        state = BoosterState(x=0.0, y=5.0, theta=0.1)
        segments = r._legSegments(state)
        plus, minus = legToes(state, world)
        toeScreens = {worldToScreen(plus, world), worldToScreen(minus, world)}
        segToes = {seg[1] for seg in segments}
        assert segToes == toeScreens
    finally:
        r.close()


def test_legSegmentsTrackWorldLegDrop():
    import dataclasses
    from src.env.physics import legToes
    base = loadConfig('config.yaml').world
    world = dataclasses.replace(base, legDrop=base.legDrop + 1.5)   # different drop
    r = Renderer(world)
    try:
        state = BoosterState(x=0.0, y=8.0, theta=0.0)
        segments = r._legSegments(state)
        plus, minus = legToes(state, world)
        toeScreens = {worldToScreen(plus, world), worldToScreen(minus, world)}
        segToes = {seg[1] for seg in segments}
        assert segToes == toeScreens
    finally:
        r.close()
