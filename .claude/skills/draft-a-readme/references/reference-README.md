<div align="center">

# Project Ordo Astra

A genetic algorithm for the Euclidean 2D Traveling Salesman Problem, framed as space exploration—every node is a planet, moon, or landmark, so each tour is an optimized exploration route.

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A52.4.6-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-%E2%89%A50.65.1-00A3E0?style=for-the-badge&logo=numba&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%E2%89%A51.14.0-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%E2%89%A53.11.0-11557C?style=for-the-badge&logo=python&logoColor=white)
![tsplib95](https://img.shields.io/badge/tsplib95-%E2%89%A50.7.1-4B8BBE?style=for-the-badge)
![License](https://img.shields.io/badge/License-TBD-lightgrey?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-local-informational?style=for-the-badge)

</div>

---

## Ⅰ • Table of Contents

- [Ⅱ • Features](#ⅱ--features)
- [Ⅲ • Demonstration](#ⅲ--demonstration)
- [Ⅳ • Quick Start](#ⅳ--quick-start)
- [Ⅴ • Installation](#ⅴ--installation)
- [Ⅵ • Usage](#ⅵ--usage)
- [Ⅶ • Configuration](#ⅶ--configuration)
- [Ⅷ • Reference](#ⅷ--reference)
- [Ⅸ • License](#ⅸ--license)
- [Ⅹ • Authors](#ⅹ--authors)
- [Ⅺ • Contact](#ⅺ--contact)

<br>

## Ⅱ • Features

- **Genetic algorithm** — population, fitness, selection, order crossover (OX), and swap mutation, over integer-encoded tours.
- **Two-opt local search** — refines the cheapest ~10% of each generation plus the running elite, sharpening tours without paying to refine the whole population.
- **Elitism with convergence detection** — the best tour always survives; a run stops early once it stagnates, hits the known optimum, or reaches the generation cap.
- **JIT-accelerated kernels** — every hot path is `numba.njit` compiled (parallel, no-GIL, cached), so generations run at native speed after a one-time compile.
- **TSPLIB instances** — loads [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) `EUC_2D` instances and reports percent error against known optimal tour lengths.
- **Batch driver** — sweeps the GA across all 78 listed `EUC_2D` instances, 10 runs each, writing one parseable log per run.
- **Analysis** — turns run logs into plots: tours, convergence curves, edge heat maps, and aggregate error / computation-time scaling across instance sizes.

<br>

## Ⅲ • Demonstration

A single run prints a per-generation trace and a summary:

```text
Instance (EUC_2D): berlin52
Number of nodes (n): 52
Optimal tour distance (fitness): 7542
Compiling and initializing GA...

Gen. Time: 0.001s, Gen.: 1, Elite: 7670.0, % Error: 1.697%
Gen. Time: 0.001s, Gen.: 2, Elite: 7542.0, % Error: 0.000%

...

Elite tour distance (fitness): 7542.0
Optimal tour distance (fitness): 7542
Percent Error (%): 0.000%
Computation Time (seconds): 0.005s
```

Then it writes a log to `stdout/runs/<instance>/<instance>_<runNum>.txt`:

```text
instanceName: berlin52
n: 52
optFit: 7542

genTime, gen, eliteFit, percentError:
8.55e-05, 0, 23237.0, 208.101
0.00171, 1, 7670.0, 1.697

...

tour:
[41 20 16  2 17 30 21  0 48 31 44 ...]

eliteFit: 7542.0
percentError: 0.0
computationTime: 0.005s
```

The analysis tools render these results into `stdout/analysis/`—final tours, per-run and aggregate convergence, edge-usage heat maps, and error / time scaling against instance size `n`.

<br>

## Ⅳ • Quick Start

```powershell
# 1. Create and activate a virtual environment named .env.local
python -m venv .env.local
.\.env.local\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full batch (78 instances x 10 runs each)
python -m src.main
```

**Note** — `python -m src.main` is a long batch job and **overwrites** existing logs in `stdout/runs/`. For a quick sanity check, run a single small instance instead (see [Usage](#ⅵ--usage)). The first GA call pays a ~10s one-time `numba` compile cost.

<br>

## Ⅴ • Installation

### Requirements

- **Python 3.14**

### Dependencies

Pinned in [requirements.txt](requirements.txt):

| Library | Version | Role |
|---------|---------|------|
| `tsplib95` | `>= 0.7.1` | Loads `.tsp` instance files |
| `numba` | `>= 0.65.1` | JIT compilation of the GA kernels |
| `numpy` | `>= 2.4.6` | Array operations and tour encoding |
| `scipy` | `>= 1.14.0` | Euclidean distance matrix (`cdist`) |
| `matplotlib` | `>= 3.11.0` | Analysis plots |

### Steps

```powershell
# 1. Clone the repository and move into it
git clone https://github.com/Kevinnnnn-ai/project-ordo-astra.git
cd project-ordo-astra

# 2. Create and activate a virtual environment named .env.local
python -m venv .env.local
.\.env.local\Scripts\Activate.ps1

# 3. Install all required dependencies
pip install -r requirements.txt
```

<br>

## Ⅵ • Usage

All commands run from the **repository root** with the project virtual environment active.

### Run the full batch

Runs every instance listed in `res/info/euc-2d-instances.txt`, `NUM_RUNS` times each:

```powershell
python -m src.main
```

### Run a single instance

To smoke-test without overwriting committed batch results, call `runGA` directly with a throwaway run number, then clean up the artifact:

```python
from src.utils.instance_util import Euc2D
from src.genetic_algorithm import runGA

runGA(Euc2D('berlin52'), 999)
```

### Generate analysis plots

The `Plot` class reads logs from `stdout/runs/` and writes images to `stdout/analysis/` (and instance scatter plots to `stdout/instance-plots/`):

```python
from src.utils.plot_util import Plot

plot = Plot()
plot.plotEuc2DInstances()              # node scatter plots for every EUC_2D instance
plot.plotTour('berlin52', runIdx=0)    # final tour for one run
plot.plotConvergence('berlin52', 0)    # percent-error vs generation for one run
plot.plotAggConvergence()              # aggregate convergence across all runs
plot.plotHeatMap('berlin52')           # edge-usage frequency across that instance's runs
plot.plotComputationTimeVsN()          # computation time vs n, across instances
plot.plotPercentErrorVsN()             # percent error vs n, across instances
```

### Regenerate the instance list

`Filter` scans the instance directory and rewrites `res/info/euc-2d-instances.txt` with only the `EUC_2D` instances:

```python
from src.utils.filter_util import Filter

Filter().filterEuc2D()
```

<br>

## Ⅶ • Configuration

GA hyperparameters are module-level constants at the top of [src/genetic_algorithm/genetic_algorithm.py](src/genetic_algorithm/genetic_algorithm.py):

| Constant | Default | Meaning |
|----------|---------|---------|
| `POP_SIZE` | `200` | Number of tours per generation |
| `MAX_GENS` | `1000` | Hard cap on generations per run |
| `SELECTION_SIZE` | `7` | Tournament size for parent selection |
| `OX_RATE` | `0.85` | Order-crossover probability |
| `SWAP_MUT_RATE` | `0.03` | Swap-mutation probability |
| `MIN_CHANGE` | `1e-3` | Minimum fitness gain to count as an improvement |
| `CONVERGENCE_GEN` | `50` | Stagnant generations before early stop |
| `TWO_OPT_PERCENTILE` | `10` | Cheapest fraction (%) of tours refined by two-opt |

Numba behavior is controlled by `NO_GIL`, `CAN_PARALLEL`, and `CAN_CACHE` in the same file.

Batch driver settings live in [src/main.py](src/main.py):

| Constant | Default | Meaning |
|----------|---------|---------|
| `INSTANCE_TYPE` | `'euc_2d'` | Instance family to run |
| `NUM_RUNS` | `10` | Runs per instance |
| `START_RUN` | `0` | Starting run index (offsets log filenames) |
| `EXCLUSIONS` | `set()` | Instance names to skip from the list |

Plot styling toggles (fit modes, aggregate-curve display) are attributes set in `Plot.__init__` in [src/utils/plot_util.py](src/utils/plot_util.py).

<br>

## Ⅷ • Reference

### Project layout

```text
project-ordo-astra/
├─ src/                          # package root (run as a module: python -m src.main)
│  ├─ main.py                    # batch driver: Run class + getInstanceNames
│  ├─ genetic_algorithm/
│  │  ├─ genetic_algorithm.py    # GA core + numba kernels; public runGA(instance, runNum)
│  │  └─ __init__.py             # re-exports runGA
│  └─ utils/
│     ├─ instance_util.py        # Euc2D loader (tour data + optimal fitness)
│     ├─ log_util.py             # Log: per-run log format (a parsing contract)
│     ├─ filter_util.py          # Filter: rebuild the EUC_2D instance list
│     └─ plot_util.py            # Plot: tours, convergence, heat maps, scaling
├─ res/
│  ├─ tsplib/tsp/                # TSPLIB .tsp instance files
│  └─ info/
│     ├─ euc-2d-instances.txt    # the EUC_2D instances the driver runs
│     └─ opt-fits.txt            # known optimal tour length per instance
└─ stdout/
   ├─ runs/                      # per-run logs (do not overwrite committed results)
   ├─ instance-plots/            # node scatter plots per instance
   └─ analysis/                  # generated analysis figures
```

### Key entry points

- **`runGA(instance, runNum)`** — runs one GA on one `Euc2D` instance, logs every generation, and returns the elite fitness.
- **`Euc2D(instanceName)`** — loads `res/tsplib/tsp/<name>.tsp` and the matching optimal fitness; exposes `getNodes()` and `getN()`.
- **`Log`** — writes the per-run log. Its format is a contract the `Plot` parsers depend on (`n:`, `computationTime:`, `percentError:`, `tour:` lines); changing it requires updating the parsers.

### Algorithm at a glance

1. Build a random initial population of tours.
2. Each generation: tournament-select parents → order crossover → swap mutation → two-opt the cheapest `TWO_OPT_PERCENTILE`%.
3. Carry the elite tour forward unchanged (elitism) and refine it with two-opt.
4. Stop when stagnation reaches `CONVERGENCE_GEN`, the optimum is hit, or `MAX_GENS` is reached.

### External

- [TSPLIB95 instance library](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) — source of the benchmark instances and the `EUC_2D` format.

<br>

## Ⅸ • License

No license file is currently distributed with this project. Until a `LICENSE` is added, all rights are reserved by the author—please contact the maintainer before reuse or redistribution.

<br>

## Ⅹ • Authors

- **Kevinnnnn-ai** — author and maintainer ([github.com/Kevinnnnn-ai](https://github.com/Kevinnnnn-ai))

<br>

## Ⅺ • Contact

- **Repository** — [github.com/Kevinnnnn-ai/project-ordo-astra](https://github.com/Kevinnnnn-ai/project-ordo-astra)
- **Issues** — please open a [GitHub issue](https://github.com/Kevinnnnn-ai/project-ordo-astra/issues) for bugs, questions, or feature requests

<br>

---

*Last Updated: June 17, 2026*
