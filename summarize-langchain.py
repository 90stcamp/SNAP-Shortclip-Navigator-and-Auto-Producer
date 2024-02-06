from datasets import load_dataset
import pandas as pd
from rouge import Rouge
from openai import OpenAI
import json
from settings import *
from utils import preprocess, prompt, score, utils
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


def llm(doc):
    template = """
    <s>[INST]<>You are an extractive summarizer that follows the output pattern.
    The following examples are successful extractive summarization instances: 

    Example Document: 'the three methods of assesing glycemic control , hba1c , smbg and cgms provide distinct information , yet complementary at the same time . hba1c does not equally reflect the glycemic values over the three months that forego its determination . \n hba1c assesses the average glycemic exposure in time without being able to differentiate between preprandial and postprandial glycemia , possible hypo and hyper glycemia . \n this method is able to identify both hypoglycemic and hyperglycemic episodes allowing immediate therapeutic decisions and therefore a glycemic balance closer to normal . \n introduction of cgms in the assessment of the glycemic status represents a great technological advance . \n this glucose monitoring method clears glycemic balance abnormalities in an otherwise impossible to obtain manner , evaluating both therapeutic efficiency and glycemic control . even if cgm systems are far from being implemented at a large scale in current practice , they are about to change the diabetes management by providing an optimal glycemic control .'
    Example Summary: 'type 2 diabetes is a chronic disease and maintaining a tight glycemic control is essential to prevent both microvascular and macrovascular complications , as demonstrated in previous studies . \n it is essential to monitor the glucose levels in order to achieve the targets . \n the blood glucose monitoring can be done by different methods : glycated haemoglobin a1c , self - monitoring of blood glucose ( before and after meals ) with a glucometer and continuous glucose monitoring with a system that measures interstitial glucose concentrations . even though glycated haemoglobin a1c is considered the  gold standard  of diabetes care \n , it does not provide complete information about the magnitude of the glycemic disequilibrium . \n therefore the self - monitoring and continuous monitoring of blood glucose are considered an important adjunct for achieving and maintaining optimal glycemic control . \n the three methods of assessing glycemic control : hba1c , smbg and cgms provide distinct but at the same time complementary information ,'
    
    Please summarize the following document.The summary should contain 3 sentences.
    Original Document: {document}<>[/INST]<\s>.
    """
    prompt = PromptTemplate(template=template, input_variables=["document"])
    llm_chain = LLMChain(prompt=prompt, llm=hf)
    response = llm_chain.invoke(input = doc)
    torch.cuda.empty_cache()
    return response['text']


def get_summarization(df,save_name, lower, upper, iter_num = 5):
    for i in range(iter_num):
        response_list = []
        for idx in tqdm(range(len(df)), total = len(df)):
            response = llm(df.iloc[idx,0])
            
            if len(response) > 0:
                response_list.append([response, df.iloc[idx, 1]])
        df = pd.DataFrame(response_list, columns = ['generate', 'abstract'])
        df.to_csv(os.path.join(OUT_DIR, f"{save_name}_{lower}_{upper}_{i}.csv"), index = False)


def get_score(save_name,lower,upper, n):
    model_avg_rouge = score.get_rouge_list_from_all_df(save_name)
    print(model_avg_rouge)
    score.save_rouge_avg(model_avg_rouge, f'{save_name}_{lower}_{upper}_{n}')
    score.statistic_from_rouge_list(f'{save_name}_{lower}_{upper}_{n}_result.npy')


if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_num', required=True, type=int
        )
    parser.add_argument(
        '--lower', required=True, type=int
        )
    parser.add_argument(
        '--upper', required=True, type=int
        )
    parser.add_argument(
        '--sample_n', required=True, type=int
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
    cache_dir = os.path.join(MODEL_DIR, args.save_name)

    # lower and upper bound for text length
    lower,upper=args.lower, args.upper
    n = args.sample_n
    iter = args.iter_n
    save_name = args.save_name
    dataset = utils.load_data(config['data_name'],lower = lower, upper = upper)
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
    result_df = get_summarization(sample, save_name, lower, upper, iter)
    get_score(save_name, lower, upper, n)
