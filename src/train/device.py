# src/train/device.py
# <agent_context>
#   [ARCH]: One source of truth for device selection — GPU-primary with an
#           automatic CPU fallback. The training stack has no other device
#           logic; every site that needs a torch.device calls resolveDevice.
#   [GOTCHA]: 'auto' (the config default) picks cuda when torch.cuda.is_available()
#             else cpu. 'cpu' forces CPU even on a GPU box (pin a run to CPU
#             without a code edit). Device is a TRAINING concern, never a world
#             one — it does NOT enter computeWorldHash, so CPU- and GPU-trained
#             checkpoints stay interchangeable (same world hash).
# </agent_context>
#
# <agent_guardrail>
#   [CRITICAL]: Keep this the ONLY place that inspects torch.cuda.is_available().
#               Scattering device probes reintroduces the CPU-only-on-a-GPU-box
#               bug this module exists to prevent.
#   [VALIDATION]: python -m pytest tests/test_device.py -v
# </agent_guardrail>
"""GPU-primary device selection with automatic CPU fallback."""
from __future__ import annotations

import torch


# @ANCHOR[resolve-device]: single device-selection entry point.
def resolveDevice(prefer: str = 'auto') -> torch.device:
    """Return the torch.device the training stack should run on.

    prefer == 'auto' (default) -> cuda if a CUDA device is present, else cpu.
    prefer == 'cpu'            -> cpu always (force the fallback path).
    Any other value raises — fail fast rather than silently guess."""
    # @INVARIANT: prefer is one of {'auto', 'cpu'}; validated at config load too.
    if prefer not in ('auto', 'cpu'):
        raise ValueError(f"device must be 'auto' or 'cpu', got {prefer!r}")
    if prefer == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda')
