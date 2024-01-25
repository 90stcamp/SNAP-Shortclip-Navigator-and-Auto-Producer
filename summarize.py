import pickle as pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm

from utils.score import *
from utils.preprocess import *
from utils.prompt import *
from utils.utils import *


def generate_output(input_text):
    # 모델에 따라 맞게 변형해서 사용하기
    inputs_idx = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    output = model.generate(inputs_idx, max_length=3000, num_return_sequences=1).to(device)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


if __name__ == '__main__':
    config = load_json(CONFIG_DIR)
    torch.cuda.empty_cache()
    dataset = load_dataset(config['data_path'])

    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    cache_dir = "/data/ephemeral/Youtube-Short-Generator"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")

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
    print('total length :', len(list_array))
    ##########################

    for idx in tqdm(range(3200,3500)):
        article, abstract = list_array[idx]
        list_article.append(article)
        list_abstract.append(abstract)

        # 문장 수 반영없는 프롬프트
        output=generate_output(prompt_basic1(list_article[-1]))

        list_generated.append(output)

    df = pd.DataFrame({'article': list_article, 'abstract': list_abstract, 'generated': list_generated})
    df.to_pickle("result.pkl")

