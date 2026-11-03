"""Validate the binary-package architecture before importing PyTorch."""

import platform
import sys


def validate_platform(system=None, machine=None, bits=None):
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    bits = (64 if sys.maxsize > 2**32 else 32) if bits is None else bits
    if bits != 64:
        raise ValueError("A 64-bit Python installation is required; see RUNNING.md")
    if system == "Darwin" and machine.lower() != "arm64":
        raise ValueError("This release requires native arm64 Python on an Apple Silicon Mac. "
                         "Intel Macs are not supported by the torch>=2.6 binary requirements. "
                         "On an M-series Mac, use a native Terminal/Python rather than Rosetta. See RUNNING.md")
    return {"system": system, "machine": machine, "python_bits": bits,
            "python": platform.python_version()}
