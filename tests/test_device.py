# tests/test_device.py
import pytest
import torch

from src.train.device import resolveDevice


def test_cpuPreferenceAlwaysCpu():
    # 'cpu' forces the fallback even on a CUDA box.
    assert resolveDevice('cpu') == torch.device('cpu')


def test_autoMatchesCudaAvailability():
    # 'auto' tracks torch.cuda.is_available(): cuda when present, cpu otherwise.
    # Asserted against the live machine so the test passes on BOTH a GPU box and
    # the CPU-only fallback path.
    expected = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert resolveDevice('auto').type == expected.type


def test_defaultIsAuto():
    assert resolveDevice().type == resolveDevice('auto').type


def test_unknownPreferenceRaises():
    with pytest.raises(ValueError):
        resolveDevice('gpu')
