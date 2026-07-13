#!/usr/bin/env python3
"""Compatibility entry point for src/mamba/prepare_hieramamba_smoke.py."""

from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "src" / "mamba" / "prepare_hieramamba_smoke.py",
    run_name="__main__",
)
