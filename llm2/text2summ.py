from utils import prompts, llmUtils, scores
import argparse 
import warnings
warnings.filterwarnings("ignore")
from contextlib import contextmanager
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import torch
import numpy as np


np.random.seed(2024)


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
        chunk_overlap = 100,
        # length_function = my_tokenizer_func,
    )
    chain = create_map_reduce_chain(prompt, hf)
    split_docs = text_splitter.create_documents([text])
    # map reduce 과정 수행
    summarize_text = chain.invoke(split_docs)
    return summarize_text['output_text']

def summarize_mapreduce(input, prompt):
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    cache_dir = "models"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, 
        max_new_tokens=400, device = 0, pad_token_id=tokenizer.eos_token_id)
    hf = HuggingFacePipeline(pipeline=pipe)

    response = get_sum(input, prompt, hf)
    
    torch.cuda.empty_cache()
    return response
