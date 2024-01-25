import pickle as pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
import json

from score import *
from preprocess import *


def make_prompt_with_sen(title, num_sen):
    # 원래 요약문 문장 수 반영한 프롬프트
    prompt = f"""
    script: {title} /n/n

    Please extract summary based on the document. The summary of the document should be 
    extracted to senteces inside the document within {num_sen} sentences. Document: [{title}] Summary: [Summary]

    """
    return prompt



def make_prompt(title):
    # 원래 요약문 그대로 반영한 프롬프트
    prompt = f"""
    script: {title} /n/n

    Please extract summary based on the document. The summary of the document should be 
    extracted to senteces inside the document. Document: [{title}] Summary: [Summary]

    """
    return prompt


def generate_output(input_text):
    # 모델에 따라 맞게 변형해서 사용하기
    inputs_idx = tokenizer.encode(input_text, return_tensors='pt')
    
    output = model.generate(inputs_idx, max_length=3000, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


if __name__ == '__main__':
    torch.cuda.empty_cache()
    dataset = load_dataset('ccdv/pubmed-summarization')

    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    cache_dir = "/data/ephemeral/Youtube-Short-Generator"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    model.to(device)

    list_article, list_abstract, list_generated = [], [], []
    list_array=[]
    ### length 순으로 sort, list_array는 임시 array
    list_array=[]
    for idx in range(len(dataset['train'])):
        article, abstract = dataset['train'][idx].values()
        list_array.append([article, abstract])
    list_array.sort(key=lambda x: len(x[0]))
    ##########################

    for idx in tqdm(range(3200,3500)):
        article, abstract = list_array[idx]
        list_article.append(article)
        list_abstract.append(abstract)

        # 프롬프트에 원래 몇 문장인지 체크하고 n문장으로 생성해달라고 반영하기
        num_sen = get_sententence_num(list_abstract[-1])
        output = generate_output(make_prompt_with_sen(list_article[-1], num_sen))

        # 문장 수 반영없는 프롬프트
        # output=generate_output(make_prompt(list_article[idx]))

        list_generated.append(output)

    df = pd.DataFrame({'article': list_article, 'abstract': list_abstract, 'generated': list_generated})
    df.to_pickle("result.pkl")

