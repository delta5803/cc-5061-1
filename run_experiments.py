#!/usr/bin/env python3
"""Generate all synthetic sequences and solve every adjacent pair for real.

Usage:
    python3 run_experiments.py [output_dir]

Default output: experiments/output
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denseflow.runner import run_all  # noqa: E402


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "experiments/output"
    manifest = run_all(out)
    print(f"\nManifest: {os.path.join(out, 'manifest.json')}")
    for s in manifest["sequences"]:
        print(f"  {s['name']:<18s} frames={s['n_frames']:2d} pairs={s['n_pairs']}")
