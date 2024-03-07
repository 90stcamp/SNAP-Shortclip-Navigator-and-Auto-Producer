from text2summ import *


def domain_entertainment(text):
    pick=summarize_langchain({"document": text}, prompts.prompt_entertain_pick())['text']
    output=summarize_langchain({"document": text, "pick":pick }, prompts.prompt_retrieval())
    return output['text']

def domain_comedy(text):
    pick=summarize_langchain({"document": text}, prompts.prompt_comedy_pick())['text']
    output=summarize_langchain({"document": text, "pick":pick }, prompts.prompt_retrieval())
    return output['text']

def domain_sports(text):
    pick=summarize_langchain(text, prompts.prompt_sports_pick())['text']
    output=summarize_langchain({"document": text, "pick":pick }, prompts.prompt_retrieval())
    return output['text']

def domain_game(text):
    pick=summarize_langchain(text, prompts.prompt_game_pick())['text']
    output=summarize_langchain({"document": text, "pick":pick }, prompts.prompt_retrieval())
    return output['text']

def separate_to_domain(domain, text):
    if domain=="Entertainment":
        return domain_entertainment(text)
    elif domain=="Sports":
        return domain_sports(text)
    elif domain=="Comedy":
        return domain_comedy(text)
    elif domain=="Gaming":
        return domain_game(text)

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
        raise Exception('domain does not match')
    return summarize_mapreduce(text, prompt)
