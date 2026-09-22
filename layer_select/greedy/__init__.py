"""Greedy plugins that turn a search space plus scores (or live eval) into slots."""

from layer_select.greedy.rank_fill import rank_fill
from layer_select.greedy.sequential import run_sequential

ALGOS = {
    "rank_fill": rank_fill,
    "sequential": run_sequential,
}


def get_algo(name: str):
    try:
        return ALGOS[name]
    except KeyError as error:
        known = ", ".join(sorted(ALGOS))
        raise ValueError(f"Unknown greedy algo {name!r}. Known: {known}") from error
