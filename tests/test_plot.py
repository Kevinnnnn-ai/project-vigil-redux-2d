# tests/test_plot.py
import os

from src.metrics.plot import plotConvergence


def _history(rates):
    return [{'iter': i, 'successRate': r} for i, r in enumerate(rates)]


def test_plotConvergenceWritesPng(tmp_path):
    outPath = str(tmp_path / 'convergence.png')
    histories = {
        1: _history([0.1, 0.3, 0.6, 0.8]),
        2: _history([0.0, 0.2, 0.5, 0.7]),
    }
    plotted = plotConvergence(histories, outPath, rolloutSteps=64, numEnvs=2)
    assert plotted == [1, 2]
    assert os.path.exists(outPath)
    assert os.path.getsize(outPath) > 0          # non-empty PNG


def test_plotConvergenceSkipsStubSeed(tmp_path):
    outPath = str(tmp_path / 'convergence.png')
    histories = {
        0: _history([0.1]),                      # stub: single point -> skipped
        1: _history([0.1, 0.4, 0.9]),
    }
    plotted = plotConvergence(histories, outPath, rolloutSteps=64, numEnvs=2)
    assert plotted == [1]
    assert os.path.exists(outPath)


def test_plotConvergenceDropsSentinelIters(tmp_path):
    # Curriculum logs successRate only on eval iters; non-eval iters carry -1.0.
    # Those sentinels must be filtered (else the line plunges to the floor).
    outPath = str(tmp_path / 'convergence.png')
    histories = {
        1: _history([0.2, -1.0, -1.0, 0.5, -1.0, -1.0, 0.9]),   # 3 real eval points
        2: _history([0.1, -1.0, -1.0, -1.0]),                   # 1 real point -> skipped
    }
    plotted = plotConvergence(histories, outPath, rolloutSteps=64, numEnvs=2)
    assert plotted == [1]
    assert os.path.exists(outPath)
