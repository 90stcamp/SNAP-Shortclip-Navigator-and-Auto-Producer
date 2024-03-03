from text2summ import *


def domain_entertainment(text):
    pick=summarize_langchain({"document": text}, prompts.prompt_entertain_pick())['text']
    output=summarize_langchain({"document": text, "pick":pick }, prompts.prompt_retrieval())
    return output['text']

def domain_comedy(text):
    pick=summarize_langchain({"document": text}, prompts.prompt_comedy_pick())['text']
    output=summarize_langchain({"document": text, "pick":pick }, prompts.prompt_retrieval())
    print(output)
    return output['text']

def domain_sports(text):
    output=summarize_langchain(text, prompts.prompt_domain_summ_entertain())
    return output['text']

def separate_to_domain(domain, text):
    if domain=="Entertainment":
        return domain_entertainment(text)
    elif domain=="Sports":
        return domain_sports(text)
    elif domain=="Comdey":
        return domain_comedy(text)
