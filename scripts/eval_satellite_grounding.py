#!/usr/bin/env python3
"""Stable entry point for the Round-2 paired satellite-grounding diagnostic.

See :mod:`scripts.eval_icassp27_sat_ablate` for the implementation and CLI.
It evaluates real, zero, cross-window visual shuffle, spatial-PE permutation,
and 90-degree RGB/geometry mismatch under paired AR sampling.
"""
from scripts.eval_icassp27_sat_ablate import main


if __name__ == "__main__":
    main()
