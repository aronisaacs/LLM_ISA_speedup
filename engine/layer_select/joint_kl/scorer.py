"""KL(dense || compressed) of the next-token distribution on calibration examples, one device.

A candidate map goes through the same path as a task run: ``kv_for_assignment`` -> ``parse_kv_spec``
-> ``install`` (the patched ``Cache.update``). One prefill forward per example and map, with the
cache on, so the prompt attends to compressed K/V as in a task run. Each example's dense forward
is shared by all candidates. Only the scored positions (the tail) are projected to logits.

Per example: mean over its scored tokens. Per domain: mean over examples (the two-level mean keeps
long answers from dominating). This module returns per-domain sums and counts, so shards on
several devices can be added up.
"""

from __future__ import annotations

from types import SimpleNamespace

from engine.kv_compress.install import install
from engine.kv_compress.spec import KvSpec, parse_kv_spec
from engine.layer_select.apply import kv_for_assignment, method_template


class KLScorer:
    def __init__(self, rows: list[dict], meta: dict, method_kv: dict, device: str,
                 model: str | None = None, dtype: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device
        name = model or meta["model"]
        kwargs = {"revision": meta["revision"]} if name == meta["model"] and meta.get("revision") else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=getattr(torch, dtype or meta["dtype"]), **kwargs).to(device).eval()
        self.lm = SimpleNamespace(model=self.model)  # what install() expects of an lm-eval wrapper
        self.template = method_template(method_kv)
        self.examples = []
        for row in rows:
            prompt, ref = row["prompt_token_ids"], row["reference_token_ids"]
            ids = torch.tensor([(prompt + ref)[:-1]], device=device)  # position i predicts token i + 1
            self.examples.append((row["domain"], ids, len(ref)))

    def spec(self, assignment) -> KvSpec:
        return parse_kv_spec(kv_for_assignment(self.template, assignment)) if assignment else KvSpec()

    def logprobs(self, ids, scored: int, spec: KvSpec):
        """Log-softmax over the last ``scored`` positions, float32, ``[scored, vocab]``."""
        torch = self.torch
        uninstall = install(self.lm, spec)
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=ids, use_cache=True, logits_to_keep=scored).logits[0]
        finally:
            uninstall()
        return torch.log_softmax(logits.float(), dim=-1)

    def kl_sums(self, candidates: dict[str, dict]) -> dict[str, dict[str, list[float]]]:
        """{key: {domain: [sum of per-example mean KL, examples]}} for {key: assignment}."""
        specs = {key: self.spec(assignment) for key, assignment in candidates.items()}
        sums: dict[str, dict[str, list[float]]] = {key: {} for key in specs}
        for domain, ids, scored in self.examples:
            dense = self.logprobs(ids, scored, KvSpec())
            dense_p = dense.exp()
            for key, spec in specs.items():
                sparse = self.logprobs(ids, scored, spec)
                kl = float((dense_p * (dense - sparse)).sum(-1).mean())
                entry = sums[key].setdefault(domain, [0.0, 0])
                entry[0] += kl
                entry[1] += 1
                del sparse
            del dense, dense_p
        return sums


def merge(parts: list[dict[str, dict[str, list[float]]]]) -> dict[str, dict[str, float]]:
    """Add per-shard sums and counts, then divide: {key: {domain: example-mean KL}}."""
    total: dict[str, dict[str, list[float]]] = {}
    for part in parts:
        for key, domains in part.items():
            for domain, (s, n) in domains.items():
                entry = total.setdefault(key, {}).setdefault(domain, [0.0, 0])
                entry[0] += s
                entry[1] += n
    return {key: {d: s / n for d, (s, n) in sorted(domains.items())} for key, domains in total.items()}
