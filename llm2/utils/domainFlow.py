from text2summ import *
from utils import prompts


def separate_reduce_domain(domain, text):
    if domain=="Entertainment":
        prompt=prompts.prompt_reduce_entertain_pick()
    elif domain=="News \\u0026 Politics":
        prompt=prompts.prompt_reduce_news_pick()
    elif domain=="Comedy":
        prompt=prompts.prompt_reduce_comedy_pick()
    elif domain=="Science \\u0026 Technology":
        prompt=prompts.prompt_reduce_sci_pick()
    elif domain=="Travel \\u0026 Events":
        prompt=prompts.prompt_reduce_travel_pick()
    else:
        prompt=prompts.prompt_reduce_entertain_pick()
    return summarize_mapreduce(text, prompt)
