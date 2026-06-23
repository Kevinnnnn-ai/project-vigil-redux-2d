# src/metrics/plot.py
# <agent_context>
#   [ARCH]: Renders a training-convergence figure (eval success rate vs. cumulative
#           environment steps) from in-memory per-iteration histories, one line per
#           seed overlaid. Called at the end of a training session to store a plot
#           alongside the run's checkpoints in checkpoints/run-N/.
#   [API]:  plotConvergence(histories, outPath, rolloutSteps, numEnvs, title=None).
#   [GOTCHA]: Uses the headless 'Agg' backend (set BEFORE importing pyplot) so no
#             window pops up mid/post training; the caller wraps this in try/except
#             so a plotting failure never discards a completed training run.
#   [GOTCHA]: Per-seed curves are monotone-cubic SMOOTHED for display (_monotoneSmooth);
#             the true eval points are over-plotted as markers so smoothing never hides
#             where real measurements were taken. The x-axis offset is pinned to 1e6.
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: x-axis = iter * rolloutSteps * numEnvs. These must be the SAME values
#               training actually ran with (cfg.training.{rolloutSteps,numEnvs}) or the
#               step axis is wrong. The y-series is 'successRate' (eval landing rate),
#               NOT episode reward — the CSV/history logs no reward channel.
#   [CRITICAL]: Smoothing is shape-preserving (PCHIP/monotone) ON PURPOSE — it must not
#               overshoot outside the data, or success-rate curves would render points
#               below 0 / above 1 that were never measured. Do not swap in a plain cubic.
#   [VALIDATION]: python -m pytest tests/test_plot.py -v
# </agent_guardrail>
"""plotConvergence: overlay eval success-rate curves per seed vs. env steps."""
from __future__ import annotations

import numpy as np

# @DEP[→agg-backend]: select headless backend before pyplot binds a GUI one.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# @CONFIG[x-axis-exponent]: pin the x-axis scientific offset to 1e6 so the step axis
# always reads in millions, instead of auto-upgrading to 1e7/1e8 on longer runs. The
# ScalarFormatter fixes the exponent when scilimits low == high != 0 (a public path —
# no deprecated orderOfMagnitude attribute).
_X_AXIS_EXPONENT = 6
# @CONFIG[smooth-samples]: dense resample count for the per-seed curves. Eval points
# are sparse, so a few hundred samples render a smooth line with no visible faceting.
_SMOOTH_SAMPLES = 240


def _sign(v):
    """int sign in {-1, 0, +1}; int() casts dodge numpy bool subtraction."""
    return int(v > 0) - int(v < 0)


def _endpointSlope(h0, h1, d0, d1):
    """PCHIP non-centered endpoint tangent (Fritsch–Carlson), shape-limited so the end
    segment never introduces an extremum the data itself does not have."""
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if _sign(m) != _sign(d0):                               # opposite the secant -> flatten
        return 0.0
    if _sign(d0) != _sign(d1) and abs(m) > 3.0 * abs(d0):   # would overshoot -> clamp
        return 3.0 * d0
    return m


def _monotoneSmooth(x, y, samples=_SMOOTH_SAMPLES):
    """Dense shape-preserving (monotone cubic / PCHIP) resample of the polyline (x, y),
    so the sparse eval points render as a smooth curve instead of blocky straight
    segments. Monotone => no overshoot beyond the data on any interval, so success-rate
    curves stay within their own [min, max] between knots (never dip < 0 or > 1). The
    curve passes through every original point. Returns the raw points unchanged when
    there are < 3 of them (two points are already a straight line; nothing to smooth)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return x, y
    h = np.diff(x)
    delta = np.diff(y) / h
    # interior tangents: 0 at local extrema / sign flips, else the weighted harmonic mean
    m = np.zeros(n)
    for k in range(1, n - 1):
        if _sign(delta[k - 1]) * _sign(delta[k]) <= 0:
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    m[0] = _endpointSlope(h[0], h[1], delta[0], delta[1])
    m[-1] = _endpointSlope(h[-1], h[-2], delta[-1], delta[-2])
    # vectorized cubic-Hermite evaluation on a dense uniform grid
    xs = np.linspace(x[0], x[-1], samples)
    idx = np.clip(np.searchsorted(x, xs) - 1, 0, n - 2)
    hk = h[idx]
    t = (xs - x[idx]) / hk
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    ys = (h00 * y[idx] + h10 * hk * m[idx]
          + h01 * y[idx + 1] + h11 * hk * m[idx + 1])
    return xs, ys


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
        # @ANCHOR[smooth-line]: the monotone-smoothed curve is the labeled line; the
        # TRUE eval points ride on top (same color, '_nolegend_' so they add no legend
        # entry) — smoothing must never hide where real measurements landed.
        xs, ys = _monotoneSmooth(steps, rates)
        line, = ax.plot(xs, ys, label=f'Seed {seed}', linewidth=1.8)
        ax.plot(steps, rates, label='_nolegend_', linestyle='none', marker='o',
                markersize=3, color=line.get_color(), alpha=0.55)
        plotted.append(seed)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Eval Success Rate')
    # @ANCHOR[y-headroom]: pad past [0,1] so curves pinned at 1.0 (or 0.0) aren't
    # clipped flush against the frame and stay visible.
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or 'Training Convergence')
    ax.grid(True, alpha=0.3)
    if plotted:
        # @ANCHOR[x-exponent]: keep the offset at 1e6 (scilimits low == high == 6) on
        # every run; only set once there is data so the empty "warming up" frame keeps
        # the default auto scale.
        ax.ticklabel_format(axis='x', style='sci',
                            scilimits=(_X_AXIS_EXPONENT, _X_AXIS_EXPONENT))
        # @ANCHOR[legend-outside]: park the legend to the RIGHT of the axes so it never
        # occludes the curves. bbox_inches='tight' + bbox_extra_artists (below) grow the
        # saved canvas to include it.
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                           borderaxespad=0.0)
    else:
        legend = None

    fig.tight_layout()
    # @SIDEFX: write PNG to disk (outPath). bbox_extra_artists keeps the outside legend
    # from being cropped by bbox_inches='tight'.
    extra = (legend,) if legend is not None else ()
    fig.savefig(outPath, dpi=120, bbox_inches='tight', bbox_extra_artists=extra)
    plt.close(fig)
    return plotted
