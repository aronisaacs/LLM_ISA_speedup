"""HumanEval with the CodeLlama body indent (see humaneval_codellama.yaml)."""

import evaluate as hf_evaluate

CODELLAMA_BODY_INDENT = "    "
compute_ = hf_evaluate.load("code_eval")


def pass_at_k(references, predictions, k=None):
    assert k is not None
    return compute_.compute(references=references, predictions=predictions,
                            k=[k] if isinstance(k, int) else k)[0]


def doc_to_text_codellama(doc):
    return doc["prompt"] + CODELLAMA_BODY_INDENT


def build_predictions_codellama(resps, docs):
    return [[doc_to_text_codellama(doc) + r for r in resp] for resp, doc in zip(resps, docs)]
