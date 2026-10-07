#!/usr/bin/env bash
# Regenerates all 8 layer maps from the WikiText tile-8 profile.
set -e
cd "$(dirname "$0")"
for scope in per-target global; do
  for b in 10 20 30 40; do
    python3 greedykv_allocation.py --input llama31_wikitext_tile8_profile.jsonl \
      --task wikitext --tile 8 --target 0.$((b/10)) --scope $scope \
      --output ../layer_maps/${scope/-/_}/${scope/-/_}_$b.json
  done
done
