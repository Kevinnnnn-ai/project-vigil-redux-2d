# src/metrics/logger.py
"""Minimal dict -> CSV logger for per-iteration training metrics. Writes a header
from the first record's keys, then one row per record. Numbers only; the caller
decides what to log."""
from __future__ import annotations

import csv


class CsvLogger:
    def __init__(self, path):
        self._file = open(path, 'w', newline='', encoding='utf-8')
        self._writer = None

    def log(self, record):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(record.keys()))
            self._writer.writeheader()
        self._writer.writerow(record)
        self._file.flush()

    def close(self):
        self._file.close()
