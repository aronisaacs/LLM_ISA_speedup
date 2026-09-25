"""Turn WikiText sweep scores into a mixed-level layer assignment.

  python scripts/select_layers.py --scores results/sweep --budget 0.5 \\
      --out results/sweep/selected.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.eval_runner.progress import say
from engine.layer_select.apply import kv_for_assignment, method_template
from engine.layer_select.greedy.rank_fill import rank_fill
from engine.layer_select.levels import LEVELS
from engine.layer_select.scores import kv_from_payload, load_sweep_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget",
        type=float,
        default=0.5,
        help="Mean compression across key and value slots in [0, 1], e.g. 0.5 = 50%% average",
    )
    parser.add_argument("--out", required=True, help="Write selected.json here")
    parser.add_argument("--scores", required=True, help="Directory of singleton sweep JSONs")
    parser.add_argument("--n-layers", type=int, default=None)
    args = parser.parse_args()

    dense_ppl, rows = load_sweep_scores(args.scores)
    n_layers = args.n_layers or _n_layers_from_rows(rows)
    assignment = rank_fill(rows, n_layers, args.budget, dense_ppl=dense_ppl)
    template = method_template(kv_from_payload(json.loads(Path(rows[0].path).read_text())))
    payload = _selected_payload(
        budget=args.budget,
        n_layers=n_layers,
        assignment=assignment,
        method_kv=template,
        dense_ppl=dense_ppl,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    say(f"wrote  {out}  assignment={payload['assignment']}")


def _n_layers_from_rows(rows) -> int:
    if not rows:
        raise ValueError("no singleton scores")
    return max(row.slot.layer for row in rows) + 1


def _selected_payload(*, budget, n_layers, assignment, method_kv, dense_ppl):
    return {
        "budget": budget,
        "n_layers": n_layers,
        "levels": list(LEVELS),
        "dense_ppl": dense_ppl,
        "assignment": [
            {**slot.to_dict(), "level": pct} for slot, pct in sorted(assignment.items())
        ],
        "kv": kv_for_assignment(method_kv, assignment),
    }


if __name__ == "__main__":
    main()
