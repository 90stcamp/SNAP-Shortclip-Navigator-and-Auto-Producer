from datasets import load_dataset
import pandas as pd
from rouge import Rouge
from openai import OpenAI
import json
from settings import *
from utils import preprocess, prompts, scores, utils
import os 

from langchain.llms import OpenAI, HuggingFaceHub
# from langchain import HuggingFaceHub
from langchain.chains import LLMChain 
from langchain.prompts import load_prompt, PromptTemplate
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import warnings
warnings.filterwarnings("ignore")
import argparse 
import torch
import numpy as np                                                              
config = utils.load_json(CONFIG_DIR)
np.random.seed(2024)


def get_dataset(lower,upper):
    df = utils.load_data(config['data_name'],lower = lower, upper = upper)
    return df


def llm(len_sen,doc):
    template=prompts.prompt_extsum_paper2()
    prompt = PromptTemplate(template=template, input_variables=["len_sen","document"])
    llm_chain = LLMChain(prompt=prompt, llm=hf)
    response = llm_chain.invoke(input={"len_sen": len_sen, "document": doc})
    torch.cuda.empty_cache()
    return response['text']


def get_summarization(df,save_name, lower, upper, df_name, iter_num = 5):
    df_name = df_name.split('/')[-1]
    for i in range(iter_num):
        response_list = []
        for idx in tqdm(range(len(df)), total = len(df)):
            len_sen=preprocess.get_sententence_num(df.iloc[idx,0])
            response = llm(len_sen,df.iloc[idx,0])
            if len(response) > 0:
                response_list.append([response, df.iloc[idx, 1]])
        df = pd.DataFrame(response_list, columns = ['generate', 'abstract'])
        df.to_csv(os.path.join(OUT_DIR, f"{save_name}_{lower}_{upper}_{df_name}_{i}.csv"), index = False)


def get_score(save_name,lower,upper, df_name, n):
    df_name = df_name.split('/')[-1]
    model_avg_rouge = scores.get_rouge_list_from_all_df(save_name, lower, upper, df_name)
    print(model_avg_rouge)
    scores.save_rouge_avg(model_avg_rouge, f'{save_name}_{lower}_{upper}_{df_name}_{n}')
    scores.statistic_from_rouge_list(f'{save_name}_{lower}_{upper}_{df_name}_{n}_result.npy')


if __name__=='__main__':
    # python summarize-langchain.py -mn=0 -dn=0 -l=1000 -u=3000 -sn=1000 --save_name=llama
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_num', '-mn', required=True, type=int
        )
    parser.add_argument(
        '--data_num', '-dn', required=True, type=int
        )
    parser.add_argument(
        '--lower', '-l', required=True, type=int
        )
    parser.add_argument(
        '--upper', '-u', required=True, type=int
        )
    parser.add_argument(
        '--sample_n', '-sn', required=True, type=int
        )
    parser.add_argument(
        '--iter_n', default=3, type=int
        )
    parser.add_argument(
        '--save_name', required=True, type=str
        )
    parser.add_argument(
        '--max_new_token', default=100, type=int
        )
    args = parser.parse_args()
    
    # model_name in config.json
    MODEL_NAME = config['model_name'][args.model_num]
    # cache_dir = config['cache_dir'][args.model_num]
    cache_dir = MODEL_DIR
    # lower and upper bound for text length
    lower,upper=args.lower, args.upper
    n = args.sample_n
    iter = args.iter_n
    save_name = args.save_name

    dataset = utils.load_data(config['data_name'][args.data_num], lower = lower, upper = upper)
    if len(dataset) < n:
        n = len(dataset)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir)
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=args.max_new_token, 
        device = 0, 
        pad_token_id=tokenizer.eos_token_id)
    hf = HuggingFacePipeline(pipeline=pipe)

    sample = dataset.sample(n)
    result_df = get_summarization(sample, save_name, lower, upper, config['data_name'][args.data_num],iter)
    get_score(save_name, lower, upper, config['data_name'][args.data_num],n)
