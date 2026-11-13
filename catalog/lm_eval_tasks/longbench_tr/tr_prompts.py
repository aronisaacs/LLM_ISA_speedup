"""LongBench-E prompts as LongBench's pred.py builds them: its own templates (THUDM/LongBench,
config/dataset2prompt.json), then middle truncation to MAXLEN tokens of the Llama-3.1 tokenizer
(keep the first and last MAXLEN/2 tokens), before any chat template.

MAXLEN = 31,500: LongBench's value for 32K-context models and the default in the TurboQuant first
author's QJL code. The TurboQuant paper does not state its value; this is our best guess.
Python, not Jinja: Jinja drops a template's final newline. The dataset calls LongBench's {input} "question".
"""

from functools import lru_cache

MAXLEN = 31_500
TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"

PROMPTS = {
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": 'Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    "hotpotqa": 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    "2wikimqa": 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
    "gov_report": 'You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:',
    "multi_news": 'You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:',
    "trec": 'Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}',
    "triviaqa": 'Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}',
    "samsum": 'Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}',
    "passage_count": 'There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ',
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "lcc": 'Please complete the code given below. \n{context}Next line of code:\n',
    "repobench-p": 'Please complete the code given below. \n{context}{input}Next line of code:\n',
}


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


def truncate(prompt: str, maxlen: int = MAXLEN) -> str:
    """pred.py: tokenize (with BOS), and if longer than maxlen keep the first and last maxlen/2 tokens."""
    tokenizer = _tokenizer()
    ids = tokenizer(prompt, truncation=False).input_ids
    if len(ids) <= maxlen:
        return prompt
    half = maxlen // 2
    return (tokenizer.decode(ids[:half], skip_special_tokens=True)
            + tokenizer.decode(ids[-half:], skip_special_tokens=True))


def _prompt(task: str, doc) -> str:
    return truncate(PROMPTS[task].format(context=doc["context"], input=doc["question"]))


def qasper(doc): return _prompt("qasper", doc)
def multifieldqa_en(doc): return _prompt("multifieldqa_en", doc)
def hotpotqa(doc): return _prompt("hotpotqa", doc)
def wikimqa(doc): return _prompt("2wikimqa", doc)
def gov_report(doc): return _prompt("gov_report", doc)
def multi_news(doc): return _prompt("multi_news", doc)
def trec(doc): return _prompt("trec", doc)
def triviaqa(doc): return _prompt("triviaqa", doc)
def samsum(doc): return _prompt("samsum", doc)
def passage_count(doc): return _prompt("passage_count", doc)
def passage_retrieval_en(doc): return _prompt("passage_retrieval_en", doc)
def lcc(doc): return _prompt("lcc", doc)
def repobench_p(doc): return _prompt("repobench-p", doc)
