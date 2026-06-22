# src/metrics/plot.py
# <agent_context>
#   [ARCH]: Renders a training-convergence figure (eval success rate vs. cumulative
#           environment steps) from in-memory per-iteration histories, one line per
#           seed overlaid. Called at the end of a training session to store a plot
#           alongside the checkpoints in models/<model>/<env>/.
#   [API]:  plotConvergence(histories, outPath, rolloutSteps, numEnvs, title=None).
#   [GOTCHA]: Uses the headless 'Agg' backend (set BEFORE importing pyplot) so no
#             window pops up mid/post training; the caller wraps this in try/except
#             so a plotting failure never discards a completed training run.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: x-axis = iter * rolloutSteps * numEnvs. These must be the SAME values
#               training actually ran with (cfg.training.{rolloutSteps,numEnvs}) or the
#               step axis is wrong. The y-series is 'successRate' (eval landing rate),
#               NOT episode reward — the CSV/history logs no reward channel.
#   [VALIDATION]: python -m pytest tests/test_plot.py -v
# </agent_guardrail>
"""plotConvergence: overlay eval success-rate curves per seed vs. env steps."""
from __future__ import annotations

# @DEP[→agg-backend]: select headless backend before pyplot binds a GUI one.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# @ENTRY[plot-convergence]: history dict -> PNG written to outPath.
def plotConvergence(histories, outPath, rolloutSteps, numEnvs, title=None):
    """Plot eval success rate vs. cumulative environment steps, one overlaid line
    per seed, and save a single PNG to `outPath`.

    histories: {seed: [stat dict, ...]} as returned by trainLanding/trainCurriculum;
               each record needs 'iter' and 'successRate'. Seeds with < 2 EVAL
               records (stub/aborted runs) are skipped.
    rolloutSteps, numEnvs: per-iteration env-step factor (iter -> steps).
    Returns the list of seeds actually plotted.
    """
    stepsPerIter = rolloutSteps * numEnvs

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = []
    # @ANCHOR[per-seed-line]: one convergence curve per seed, skipping stubs.
    for seed in sorted(histories):
        history = histories[seed]
        # @INVARIANT: successRate is only measured on eval iterations; the curriculum
        # loop writes a -1.0 sentinel on the rest. Plotting the sentinels makes the
        # line plunge to the floor between evals (reads as a bar graph), so keep only
        # the real eval points and connect THOSE.
        evals = [record for record in history if record['successRate'] >= 0.0]
        if len(evals) < 2:                   # @INVARIANT: need >=2 points to draw a line
            continue
        steps = [record['iter'] * stepsPerIter for record in evals]
        rates = [record['successRate'] for record in evals]
        ax.plot(steps, rates, label=f'seed {seed}', linewidth=1.5)
        plotted.append(seed)

    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Eval success rate')
    # @ANCHOR[y-headroom]: pad past [0,1] so curves pinned at 1.0 (or 0.0) aren't
    # clipped flush against the frame and stay visible.
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or 'Training convergence')
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()

    fig.tight_layout()
    # @SIDEFX: write PNG to disk (outPath).
    fig.savefig(outPath, dpi=120)
    plt.close(fig)
    return plotted
