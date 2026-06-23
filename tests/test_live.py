# tests/test_live.py
import csv
import os

from src.metrics import live


def _writeCsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as a:
        writer = csv.DictWriter(a, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_resolveNextRunEmptyRootIsOne(tmp_path):
    assert live.resolveNextRun(str(tmp_path)) == 1


def test_resolveNextRunAbsentRootIsOne(tmp_path):
    assert live.resolveNextRun(str(tmp_path / 'nope')) == 1


def test_resolveNextRunIncrementsPastHighest(tmp_path):
    (tmp_path / 'run-1').mkdir()
    (tmp_path / 'run-3').mkdir()
    (tmp_path / 'junk').mkdir()       # non-matching dir ignored
    (tmp_path / 'run-x').mkdir()      # non-numeric suffix ignored
    assert live.resolveNextRun(str(tmp_path)) == 4


def test_pathHelpersAreRunNumbered():
    checkpoints = 'checkpoints'
    logs = os.path.join('stdout', 'logs')
    plots = os.path.join('stdout', 'convergence-plots')
    assert live.runCheckpointDir(2, checkpoints) == os.path.join(checkpoints, 'run-2')
    assert live.seedCheckpointPath(2, 0, checkpoints) == os.path.join(checkpoints, 'run-2', 'seed0.pt')
    assert live.runLogsDir(2, logs) == os.path.join(logs, 'run-2')
    assert live.seedCsvPath(2, 1, logs) == os.path.join(logs, 'run-2', 'seed1.csv')
    assert live.runPlotPath(2, plots) == os.path.join(plots, 'run-2.png')


def test_readSeedHistoriesParsesPerSeed(tmp_path):
    logsDir = str(tmp_path)
    _writeCsv(os.path.join(logsDir, 'seed0.csv'), [
        {'iter': 0, 'successRate': 0.1, 'stage': 'full'},
        {'iter': 1, 'successRate': 0.5, 'stage': 'full'},
    ])
    _writeCsv(os.path.join(logsDir, 'seed2.csv'), [
        {'iter': 0, 'successRate': -1.0, 'stage': 'hop'},   # curriculum sentinel
        {'iter': 1, 'successRate': 0.8, 'stage': 'hop'},
    ])
    histories = live.readSeedHistories(logsDir)
    assert set(histories) == {0, 2}
    assert histories[0] == [
        {'iter': 0, 'successRate': 0.1},
        {'iter': 1, 'successRate': 0.5},
    ]
    # sentinel rows are retained verbatim (the plotter filters them, not the reader)
    assert histories[2][0] == {'iter': 0, 'successRate': -1.0}


def test_readSeedHistoriesSkipsTruncatedRow(tmp_path):
    # Simulates reading a CSV mid-flush: a complete row then a half-written final line.
    logsDir = str(tmp_path)
    os.makedirs(logsDir, exist_ok=True)
    with open(os.path.join(logsDir, 'seed0.csv'), 'w', newline='', encoding='utf-8') as a:
        a.write('iter,successRate,stage\n')
        a.write('0,0.3,full\n')
        a.write('1,')                 # truncated — no successRate value yet
    histories = live.readSeedHistories(logsDir)
    assert histories[0] == [{'iter': 0, 'successRate': 0.3}]


def test_readSeedHistoriesEmptyDirIsEmpty(tmp_path):
    assert live.readSeedHistories(str(tmp_path)) == {}


def test_readSeedHistoriesAbsentDirIsEmpty(tmp_path):
    assert live.readSeedHistories(str(tmp_path / 'nope')) == {}


def test_renderConvergenceWritesPng(tmp_path):
    logsDir = str(tmp_path / 'logs')
    _writeCsv(os.path.join(logsDir, 'seed0.csv'), [
        {'iter': 0, 'successRate': 0.1},
        {'iter': 1, 'successRate': 0.4},
        {'iter': 2, 'successRate': 0.9},
    ])
    outPath = str(tmp_path / 'plots' / 'run-1.png')      # 'plots' dir does not exist yet
    plotted = live.renderConvergence(logsDir, outPath, rolloutSteps=64, numEnvs=2)
    assert plotted == [0]
    assert os.path.exists(outPath)
    assert os.path.getsize(outPath) > 0


def test_renderConvergenceEmptyStillWritesAxesPng(tmp_path):
    logsDir = str(tmp_path / 'logs')
    os.makedirs(logsDir, exist_ok=True)
    outPath = str(tmp_path / 'plots' / 'run-1.png')
    plotted = live.renderConvergence(logsDir, outPath, rolloutSteps=64, numEnvs=2)
    assert plotted == []
    assert os.path.exists(outPath)                       # warming-up frame, no lines
