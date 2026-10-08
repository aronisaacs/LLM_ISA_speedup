#!/usr/bin/env python3
"""Compatibility entry point; shared chunk evaluation lives in the engine."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine.eval_runner.chunks import (partition, evaluate_chunks, combine,
    worker_configurations, spawn_workers, identity, main)

if __name__ == "__main__":
    main()
