from utils import prompts, llmUtils, scores
from settings import *
import argparse 
import json
import warnings
warnings.filterwarnings("ignore")
import time
from contextlib import contextmanager
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

import torch
import numpy as np


config = llmUtils.load_json(CONFIG_DIR)
np.random.seed(2024)

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)


def get_dataset(lower,upper):
    df = llmUtils.load_data(config['data_name'],lower = lower, upper = upper)
    return df

def input_llm(len_sen,doc):
    template=prompts.prompt_extsum_paper2()
    prompt = PromptTemplate(template=template, input_variables=["len_sen","document"])
    llm_chain = LLMChain(prompt=prompt, llm=hf)
    response = llm_chain.invoke(input={"len_sen": len_sen, "document": doc})
    torch.cuda.empty_cache()
    return response['text']

def get_summarization_experiment(df,save_name, lower, upper, df_name, iter_num = 5):
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

def summarize_langchain(input, template):
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    cache_dir = "models"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=100, device = 0, pad_token_id=tokenizer.eos_token_id)
    hf = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(template=template, input_variables=["len_sen","document"])
    llm_chain = LLMChain(prompt=prompt, llm=hf)
    response = llm_chain.invoke(input=input)
    torch.cuda.empty_cache()
    return response

def create_map_reduce_chain(prompt, hf):
    map_template = PromptTemplate(
        template=prompts.prompt_extsum_paper2(),
        input_variables=["document"]
        )
    map_chain = LLMChain(llm=hf, prompt=map_template)

    reduce_template = PromptTemplate(
        template= prompt,
        input_variables=["ext_sum"]
        )
    reduce_chain = LLMChain(llm=hf, prompt=reduce_template) 


    # reduce_chain에 전달할 문서 정의. 여러 문서를 하나로 결합하는 역할
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, 
        document_variable_name= "ext_sum" # (reduce_template 에 정의된 변수명)
    )

    # token_max를 넘어가면 문서를 결합하거나 분할하는 역할
    reduce_documents_chain = ReduceDocumentsChain(
        # map_chain들의 결과를 결합
        combine_documents_chain=combine_documents_chain,
        # max_token이 넘어가면 분할
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=1024,
    )

    # 문서들에 체인을 매핑하여 결합하고, 그 다음 결과들을 결합
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="document", # (map_template 에 정의된 변수명)
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    return map_reduce_chain

def get_sum(text, prompt, hf):
    torch.cuda.empty_cache()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        # tokenizer = tokenizer,
        separator= '\n',
        chunk_size = 1024,
        chunk_overlap = 0,
        # length_function = my_tokenizer_func,
    )

    chain = create_map_reduce_chain(prompt, hf)
    split_docs = text_splitter.create_documents([text])
    # map reduce 과정 수행
    summarize_text = chain.invoke(split_docs)
    return summarize_text['output_text']

def summarize_mapreduce(input, prompt):
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    # cache_dir = "models"
    cache_dir = MODEL_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=400, device = 0, pad_token_id=tokenizer.eos_token_id)
    hf = HuggingFacePipeline(pipeline=pipe)

    response = get_sum(input, prompt, hf)
    
    torch.cuda.empty_cache()
    return response


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
    cache_dir = config['cache_dir'][args.model_num]

    # lower and upper bound for text length
    lower,upper=args.lower, args.upper
    n = args.sample_n
    iter = args.iter_n
    save_name = args.save_name

    dataset = llmUtils.load_data(config['data_name'][args.data_num], lower = lower, upper = upper)
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
    result_df = get_summarization_experiment(sample, save_name, lower, upper, config['data_name'][args.data_num],iter)
    get_score(save_name, lower, upper, config['data_name'][args.data_num],n)
