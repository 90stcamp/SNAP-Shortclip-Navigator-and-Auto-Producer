from utils import prompts, utils
from settings import *
import argparse 
import json
import warnings
warnings.filterwarnings("ignore")
import time
from contextlib import contextmanager

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
import numpy as np                                                              
config = utils.load_json(CONFIG_DIR)
np.random.seed(2024)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def map_prompt():
    template="""
    <s>[INST]<>Find the entertaining moment in this script and return it.

    Document: {document}<>[/INST]<\s>.
    """
    return template

def reduce_prompt():
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most entertaining moments or 'hot clips' from various parts of the script.
    Entertaining Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template


def create_map_reduce_chain():
    map_template = PromptTemplate(
        template=prompts.prompt_extsum_paper2(),
        input_variables=["document"]
        )
    map_chain = LLMChain(llm=hf, prompt=map_template)

    reduce_template = PromptTemplate(
        template= reduce_prompt(),
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
        token_max=args.chunk_size,
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

def my_tokenizer_func(text):
    return len(tokenizer.encode(text))

def get_sum(text):
    torch.cuda.empty_cache()
    text_splitter = CharacterTextSplitter(
        separator= '\n',
        chunk_size = args.chunk_size,
        chunk_overlap = args.chunk_overlap,
        length_function = my_tokenizer_func,
    )
    chain = create_map_reduce_chain()
    split_docs = text_splitter.create_documents([text])
    # map reduce 과정 수행
    summarize_text = chain.invoke(split_docs)
    return summarize_text['output_text']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text_path', required= True, type=str
        )
    parser.add_argument(
        '--model_num', default=1, type=int
        )
    parser.add_argument(
        '--max_token_num', default=400, type=int
        )
    parser.add_argument(
        '--chunk_size', default=1024, type=int
        )
    parser.add_argument(
        '--chunk_overlap', default=100, type=int
        )
    args = parser.parse_args()

    print("Load model and Setting pipline...")
    with timer('Summarization'):
        MODEL_NAME = config['model_name'][args.model_num]
        cache_dir = config['cache_dir'][args.model_num]
        # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
        # cache_dir = "../Youtube-Short-Generator/models/mistral"


        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=cache_dir)
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens= args.max_token_num,
            device = 0, 
            pad_token_id=tokenizer.eos_token_id)
        hf = HuggingFacePipeline(pipeline=pipe)
        torch.cuda.empty_cache()
        print("Load model and Setting pipline End...")

        print('Load Text file...')
        with open(args.text_path, 'r') as f:
            lines = f.readlines()
        text = ' '.join(lines)
        print('Load Text file End...')


        torch.cuda.empty_cache()
        print('Summarization Start...')
        summarize_text = get_sum(text)
        print('Summarization End...')
        with open('./summarize_result.txt', 'w') as f:
            f.write(summarize_text)
