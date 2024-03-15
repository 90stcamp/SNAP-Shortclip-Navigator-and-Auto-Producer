import openai
import json
from tqdm import tqdm
from datasets import load_dataset


from utils.score import *
from utils.preprocess import *


def get_open_ai_key():
    with open('key.json', 'r') as f:
        openai.api_key = json.load(f)['openai_key']

def prompt_system():
    system="You are helpful AI to do tasks."
    
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system}
        ],
        temperature=0,
        max_tokens=256,
    )

def prompt_user(prompt, temperature=0, max_tokens=256):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content

def make_prompt(title,num_sen):
    prompt = f"""
    script: {title}

    Please extract summary based on the document. The summary of the document should be extracted inside the document within {num_sen} sentences. Document: [{title}] Summary: [Summary]

    """
    return prompt


if __name__ == '__main__':
    get_open_ai_key()
    
    ## 데이터셋 특징에 따라 다르게 변형하기
    dataset = load_dataset('ccdv/pubmed-summarization')
    
    # prompt 임시로 지정
    prompt=make_prompt('','')
    
    ## article: 데이터셋 원문 리스트, abstract: 데이터셋 요약문 리스트 ###
    list_article,list_abstract=[],[]
    
    for idx in tqdm(range(400)):
        article,abstract=list(dataset['train'][idx].values())
        if get_token_num(article)+get_token_num(prompt)>4096:
            continue
        list_article.append(article)
        list_abstract.append(abstract)
    ####################
    
    list_generated=[]
    for idx in tqdm(range(len(list_article))):
        # 프롬프트에 원래 몇 문장인지 체크하고 n문장으로 생성해달라고 반영하기
        num_sen=get_sententence_num(list_article[idx])
        output=prompt_user(make_prompt(list_article[idx],num_sen))
        list_generated.append(output)

    result_json={}
    score=get_Rouge_score(list_generated,list_abstract)
    result_json['abstract_original']=list_abstract
    result_json['abstract_generated']=list_generated
    result_json['score']=score

    with open('result_json.json', 'w') as f:
        json.dump(result_json, f, indent=4)

