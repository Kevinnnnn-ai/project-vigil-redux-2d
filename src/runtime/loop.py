# src/runtime/loop.py
# <agent_context>
#   [ARCH]: The runtime conductor — the one place that knows the per-frame cycle
#           for BOTH runtime modes. watch and play differ only in the action
#           source they pass, so this loop has no mode flag. Reads env.state to
#           feed the renderer — render never sees the env.
#   [GOTCHA]: Pause halts env stepping but keeps drawing; stepOnce advances
#             exactly one env step while paused. autoReset restarts the episode
#             in place so watch runs forever and play immediately offers retry.
#   [GOTCHA]: lastAction persists across paused frames so the flame freezes
#             mid-burn instead of vanishing.
# </agent_context>
# <agent_guardrail>
#   [CRITICAL]: Do NOT import pygame here — the loop must stay testable with a
#               fake renderer. All windowing lives in render.py.
#   [VALIDATION]: python -m pytest tests/test_runtime_loop.py -v
# </agent_guardrail>
"""Mode-free episode loop over LandingEnv + a duck-typed renderer + one action
source: a callable (obs, intents) -> env action (2,). A net source ignores
intents; the human source reads them."""
from __future__ import annotations

import numpy as np

_SPEED_MIN = 0.125
_SPEED_MAX = 8.0
# @TAG[end-dwell]: when an episode ends, HOLD the terminal frame (the booster
# resting on its legs, or toppled) on screen for this many frames before
# auto-resetting. Without it the loop reset the env on the SAME frame it
# terminated and drew the freshly respawned booster instead — so the touchdown,
# the leg-settling steps, and the final pose were never visible (the booster
# appeared to vanish at the ground and reappear aloft). At fps=20 this is ~1.6 s.
_END_DWELL_FRAMES = 32


def runEpisodeLoop(
    env, renderer, source, rng,
    *, fps=20, autoReset=True, onEpisodeEnd=None, hudFn=None, maxFrames=None,
    endDwellFrames=_END_DWELL_FRAMES,
):
    """Drive episodes until the user quits (or maxFrames, for smoke tests).

    source: callable(obs, intents) -> action [throttle, gimbal] in env space.
    onEpisodeEnd(outcome): called with info['outcome'] at each episode end.
    hudFn(isPaused, speed) -> list[str]: HUD lines for the current frame.

    On an episode end the FINAL state (resting on its legs or toppled) is drawn
    and held for endDwellFrames before auto-reset, so the touchdown and settling
    are actually seen. The dwell respects pause/quit and can be cut short with a
    manual reset (r). Set endDwellFrames=0 to restore instant auto-reset.
    """
    obs = env.reset(rng)
    lastAction = np.zeros(2)
    isPaused = False
    speed = 1.0
    frame = 0
    while True:
        if maxFrames is not None and frame >= maxFrames:
            break
        intents = renderer.pollIntents()
        if intents.quit:
            break
        if intents.togglePause:
            isPaused = not isPaused
        if intents.speedUp:
            speed = min(speed * 2.0, _SPEED_MAX)
        if intents.speedDown:
            speed = max(speed / 2.0, _SPEED_MIN)
        if intents.reset:
            obs = env.reset(rng)
            lastAction = np.zeros(2)

        isAdvancing = (not isPaused) or intents.stepOnce
        if isAdvancing:
            action = source(obs, intents)
            obs, reward, terminated, truncated, info = env.step(action)
            lastAction = np.asarray(action, dtype=np.float64)
            if terminated or truncated:
                if onEpisodeEnd is not None:
                    onEpisodeEnd(info['outcome'])
                # Draw and HOLD the terminal frame (the rested/toppled booster)
                # before resetting, so the outcome is visible instead of being
                # skipped by an instant respawn. @TAG[end-dwell].
                frame = _dwellOnEnd(
                    env, renderer, hudFn, isPaused, speed, lastAction,
                    fps, frame, maxFrames, endDwellFrames,
                )
                if autoReset:
                    obs = env.reset(rng)
                    lastAction = np.zeros(2)
                continue

        hudLines = hudFn(isPaused, speed) if hudFn is not None else []
        renderer.draw(env.state, lastAction, hudLines)
        renderer.tick(fps * speed)
        frame += 1
    renderer.close()


def _dwellOnEnd(
    env, renderer, hudFn, isPaused, speed, lastAction,
    fps, frame, maxFrames, endDwellFrames,
):
    """Hold the just-ended terminal frame on screen for endDwellFrames, drawing
    the final (resting/toppled) state each tick. Returns the updated frame count.
    Honors quit and a manual reset (either cuts the dwell short); a pause freezes
    the held frame without consuming the dwell budget."""
    held = 0
    while held < endDwellFrames:
        if maxFrames is not None and frame >= maxFrames:
            break
        intents = renderer.pollIntents()
        if intents.quit or intents.reset:
            break
        if intents.togglePause:
            isPaused = not isPaused
        hudLines = hudFn(isPaused, speed) if hudFn is not None else []
        renderer.draw(env.state, lastAction, hudLines)
        renderer.tick(fps * speed)
        frame += 1
        if not isPaused:
            held += 1
    return frame
