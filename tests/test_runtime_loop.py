# tests/test_runtime_loop.py
import numpy as np
import pytest

from src.config.loader import loadConfig
from src.env.episode import LandingEnv
from src.agents.scripted import PdPilot
from src.runtime.render import Intents
from src.runtime.loop import runEpisodeLoop


@pytest.fixture
def cfg():
    return loadConfig('config.yaml')


class FakeRenderer:
    """Headless renderer double: feeds a scripted list of Intents and records
    every draw call. After the script is exhausted it keeps returning plain
    Intents() (no input)."""

    def __init__(self, script=()):
        self.script = list(script)
        self.draws = []
        self.isClosed = False

    def pollIntents(self):
        if self.script:
            return self.script.pop(0)
        return Intents()

    def draw(self, state, action, hudLines):
        self.draws.append((state, tuple(action), tuple(hudLines)))

    def tick(self, fps):
        pass

    def close(self):
        self.isClosed = True


def _stillSource(obs, intents):
    return np.array([0.0, 0.0])


def test_quitStopsLoopAndClosesRenderer(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[0])
    renderer = FakeRenderer(script=[Intents(), Intents(), Intents(quit=True)])
    runEpisodeLoop(env, renderer, _stillSource, np.random.default_rng(0), fps=1000)
    assert renderer.isClosed
    assert len(renderer.draws) == 2     # quit frame draws nothing


def test_pauseFreezesEnvAndStepOnceAdvances(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[-1])
    script = [
        Intents(togglePause=True),   # pause immediately (env.t stays 0)
        Intents(),                   # paused: no step
        Intents(stepOnce=True),      # exactly one step
        Intents(),                   # still paused
        Intents(quit=True),
    ]
    renderer = FakeRenderer(script=script)
    runEpisodeLoop(env, renderer, _stillSource, np.random.default_rng(0), fps=1000)
    assert env.t == 1


def test_resetRespawns(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[-1])
    script = [Intents(), Intents(), Intents(reset=True), Intents(quit=True)]
    renderer = FakeRenderer(script=script)
    runEpisodeLoop(env, renderer, _stillSource, np.random.default_rng(0), fps=1000)
    # Two free frames advanced t to 2; the reset frame re-spawns (t=0) and then
    # advances once more in the same frame.
    assert env.t == 1


def test_pdPilotLandsThroughTheLoop(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[0])
    pilot = PdPilot(cfg.world)
    outcomes = []
    renderer = FakeRenderer()
    runEpisodeLoop(
        env, renderer, lambda obs, intents: pilot.act(obs),
        np.random.default_rng(0), fps=1000,
        onEpisodeEnd=outcomes.append, maxFrames=400,
    )
    assert 'success' in outcomes        # landed at least once, autoReset continued
    assert len(renderer.draws) == 400


def test_hudFnReceivesPauseAndSpeed(cfg):
    env = LandingEnv(cfg, stage=cfg.curriculum.stages[0])
    seen = []

    def hud(isPaused, speed):
        seen.append((isPaused, speed))
        return ['hud']

    script = [Intents(speedUp=True), Intents(quit=True)]
    renderer = FakeRenderer(script=script)
    runEpisodeLoop(
        env, renderer, _stillSource, np.random.default_rng(0),
        fps=1000, hudFn=hud,
    )
    assert seen == [(False, 2.0)]
    assert renderer.draws[0][2] == ('hud',)
