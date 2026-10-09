#!/usr/bin/env python3
"""Time the parts of one group_rd compression call: menu, price search (batched vs bisection), reconstruction.

Random keys shaped like one Llama 3.1 8B slot (8 KV heads, head dim 128) at a C-Eval-like
prompt length and at 2048 tokens. Checks that both price searches choose the same formats.
Takes about a minute on one GPU; prints milliseconds per call.

  python compression_topics/spatial/scripts/group_rd_benchmark.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.scripts.group_rd_offline import CANDIDATE_MENUS
from engine.kv_compress.rope import RopeTables


def timed(function, device, repeats):
    function()  # warm up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        result = function()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) / repeats * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--lengths", type=int, nargs="+", default=[672, 2048])
    args = parser.parse_args()
    rope = RopeTables(500000., 128)
    menu = CANDIDATE_MENUS["flag_d"]
    torch.manual_seed(0)
    for length in args.lengths:
        keys = (torch.randn(1, 8, 1, 128) + torch.randn(1, 8, length, 128).cumsum(-2) * .03
                + .5 * torch.randn(1, 8, length, 128)).to(args.device)
        target = keys.numel() * 16 * .6
        table, t_menu = timed(lambda: group_rd.build_menu(keys, rope_tables=rope, menu=menu, distortion="squared"),
                              args.device, args.repeats)
        fast, t_fast = timed(lambda: group_rd.solve_lambda(table, target), args.device, args.repeats)
        slow, t_slow = timed(lambda: group_rd.solve_lambda_bisection(table, target), args.device, args.repeats)
        choice = group_rd.select(table, fast)
        same = torch.equal(choice, group_rd.select(table, slow))
        _, t_recon = timed(lambda: group_rd.reconstruct(keys, table, choice, rope), args.device, args.repeats)
        print(f"{length} tokens on {args.device}: menu {t_menu:.1f} ms, price search batched {t_fast:.1f} ms "
              f"vs bisection {t_slow:.1f} ms, reconstruct {t_recon:.1f} ms; same choices: {same}", flush=True)
    print("A C-Eval request compresses 64 slots (32 layers x keys/values), so multiply by 64 per request.")


if __name__ == "__main__":
    main()
