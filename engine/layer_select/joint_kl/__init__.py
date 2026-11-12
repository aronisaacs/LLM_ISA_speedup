"""Joint greedy allocation scored by KL divergence from the dense model on four domains.

Adapted from the E18 bundle (research partner's kv-sparsity project). Each round tries one more
rung on every eligible slot, on top of the map built so far, and keeps the slot with the smallest
increase in ``kl_mean`` per unit of saving. Lazy rounds re-score only a shortlist.

  calibration_set  the 512-example manifest, its check, and the split across GPUs
  search           the greedy itself (no torch; takes a scoring function)
  scorer           KL of a candidate map against dense on one device
  pool             one scorer per GPU, examples split between them
"""
