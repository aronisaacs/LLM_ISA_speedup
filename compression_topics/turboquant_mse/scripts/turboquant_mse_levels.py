#!/usr/bin/env python3
"""Four uniform TurboQuant levels on every layer, K and V: CEval, GSM8K and HumanEval.

  4 bits everywhere                                       4.0 bits per value
  3 bits everywhere                                       3.0 bits per value
  64 calibrated outlier channels per head at 3 bits,
  the other 64 at 2 bits                                  2.5 bits per value
  2 bits everywhere                                       2.0 bits per value

The paper's "2.5-bit" runs put 32 outliers at 3 bits and 96 channels at 2, which is
(32 x 3 + 96 x 2) / 128 = 2.25 bits, not the 2.5 it states. 64 + 64 is 2.5 bits, and
both halves are powers of two, so each uses a 64-point Hadamard (no dense rotation).
Signs: one random sign per RoPE pair (catalog default), so D can be folded into W_q / W_k.

Compression counts index bits only (16-bit dense values; fp16 norms not counted),
as the paper does. Runs the outlier calibration first when its table is missing.
Rows already in results.json (dense) are skipped.

  python compression_topics/turboquant_mse/scripts/turboquant_mse_levels.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog.compressions import TURBOQUANT_OUTLIER_TABLE as TABLE, turboquant_mse  # noqa: E402
from engine.layer_select.budgets import budget_run  # noqa: E402

OUTLIERS = 64
HEAD_DIM = 128


def levels() -> list[dict]:
    """One selection per level: tag, kv, and the index-bit compression vs 16-bit values."""
    split_bits = (OUTLIERS * 3 + (HEAD_DIM - OUTLIERS) * 2) / HEAD_DIM
    rows = [
        ("all4", turboquant_mse(bits=4), 4.0),
        ("all3", turboquant_mse(bits=3), 3.0),
        (f"out{OUTLIERS}_3b_2b", turboquant_mse(bits=2, outliers=OUTLIERS, outlier_table=TABLE), split_bits),
        ("all2", turboquant_mse(bits=2), 2.0),
    ]
    return [
        {"tag": f"turboquant_mse_{tag}", "kv": kv, "budget": 1 - bits / 16, "compression": 1 - bits / 16}
        for tag, kv, bits in rows
    ]


# Longest task first. multi_run gives each GPU every Nth configuration: with 3 GPUs and 4 levels,
# one GPU gets two GSM8K runs and the others one each. Dense rows (already recorded) go last.
TASK_ORDER = ("gsm8k", "humaneval_instruct", "ceval")


def levels_run() -> dict:
    run = budget_run(levels())
    task = lambda c: next(i for i, name in enumerate(TASK_ORDER) if f"llama31_{name}_" in c["name"])
    run["configurations"].sort(key=lambda c: (c["name"].endswith("_dense"), task(c)))
    return run


def main() -> None:
    from engine.eval_runner.progress import mirror_terminal

    from compression_topics.turboquant_mse.scripts.calibrate_outliers import ensure_table

    mirror_terminal(ROOT)
    ensure_table(TABLE)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(levels_run(), handle)
        path = Path(handle.name)
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "engine" / "multi_run.py"), "--run", str(path), "--skip-existing"],
            cwd=ROOT,
            check=True,
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
