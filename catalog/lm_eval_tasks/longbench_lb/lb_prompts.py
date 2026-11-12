"""LongBench's own prompts (THUDM/LongBench, LongBench/config/dataset2prompt.json) for the six
LongBench-E tasks whose lm-eval prompts differ from them in trailing whitespace.

Python functions, not Jinja templates: Jinja drops a template's final newline, which is how
the code prompts lost their "Next line of code:\n". The dataset calls LongBench's {input} "question".
"""

TRIVIAQA = 'Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}'


def triviaqa(doc):
    return TRIVIAQA.format(context=doc["context"], input=doc["question"])


SAMSUM = 'Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}'


def samsum(doc):
    return SAMSUM.format(context=doc["context"], input=doc["question"])


PASSAGE_COUNT = 'There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: '


def passage_count(doc):
    return PASSAGE_COUNT.format(context=doc["context"], input=doc["question"])


PASSAGE_RETRIEVAL_EN = 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '


def passage_retrieval_en(doc):
    return PASSAGE_RETRIEVAL_EN.format(context=doc["context"], input=doc["question"])


LCC = 'Please complete the code given below. \n{context}Next line of code:\n'


def lcc(doc):
    return LCC.format(context=doc["context"], input=doc["question"])


REPOBENCH_P = 'Please complete the code given below. \n{context}{input}Next line of code:\n'


def repobench_p(doc):
    return REPOBENCH_P.format(context=doc["context"], input=doc["question"])
