"""Deterministic token chunks shared by profiling and calibration."""
import random
import torch


def wikitext_chunks(tokenizer, count: int, seq_len: int = 2048, seed: int = 0, *, split='test') -> tuple[list[torch.Tensor], list[int]]:
    """Random chunks of the chosen WikiText-2 split (legacy default test), BOS plus text.

    The pages are joined and tokenized once, then cut
    into back-to-back chunks. Returns the chunks and their indices in that cut.
    """
    from datasets import load_dataset

    if split not in ('train', 'validation', 'test'):
        raise ValueError('WikiText split must be train, validation or test')
    pages = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split=split)["page"]
    tokens = tokenizer("\n\n".join(pages), add_special_tokens=False, verbose=False)["input_ids"]
    body = seq_len - 1
    available = len(tokens) // body
    if count > available:
        raise ValueError(f"WikiText-2 {split} has {available} chunks of {seq_len} tokens, asked for {count}")
    chosen = sorted(random.Random(seed).sample(range(available), count))
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    chunks = [torch.tensor(bos + tokens[i * body : (i + 1) * body]) for i in chosen]
    return chunks, chosen
