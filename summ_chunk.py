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

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def enter_reduce_prompt():
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most entertaining moments or 'hot clips' from various parts of the script.
    Entertaining Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template

def process_chunk(chunk):
    # 청크를 map_chain을 통해 처리하는 코드로 대체
    formatted_input = {"document": chunk.page_content}  # 'content'는 청크 내용을 가리키는 키
    result = map_chain.apply([formatted_input])  # 가정: map_chain.apply 메소드 사용
    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--script_name', required= True, type=str
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
        '--chunk_overlap', default=200, type=int
        )
    parser.add_argument(
        '--save_name', required=True, type=str
        )
    args = parser.parse_args()

    MODEL_NAME = config['model_name'][args.model_num]
    cache_dir = config['cache_dir'][args.model_num]
    with timer('Summarization'):
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

        map_template = PromptTemplate(
            template=prompts.prompt_entertaining_map(),
            input_variables=["document"]
            )
        map_chain = LLMChain(llm=hf, prompt=map_template)
        torch.cuda.empty_cache()

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator= '\n',
            chunk_size = 1024,
            chunk_overlap = 100,
            # length_function = my_tokenizer_func,
        )
        with open(os.path.join(SUMM_DIR, f"script/{args.script_name}.txt"), 'r') as f:
            lines = f.readlines()
        text = ' '.join(lines)
        
        split_docs = text_splitter.create_documents([text])

        chunks = split_docs  # split_docs는 텍스트를 청크로 나눈 결과 리스트
        max_workers = os.cpu_count()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            # 완료된 작업의 결과 처리
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        reduce_template = PromptTemplate(
            template=prompts.prompt_entertaining_map(),
            input_variables=["document"]
            )
        reduce_chain = LLMChain(llm=hf, prompt=reduce_template) 

        str_list = [result[0]['text'] for result in results]
        reduce_result = reduce_chain.invoke(' '.join(str_list))['text']

        with open(os.path.join(SUMM_DIR, f"result/{args.save_name}.txt"), 'w') as f:
            f.write(reduce_result)