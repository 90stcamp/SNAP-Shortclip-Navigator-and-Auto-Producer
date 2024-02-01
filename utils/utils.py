from settings import *
import json
from openai import OpenAI
from datasets import load_dataset
import pandas as pd


def load_json(path):
    with open(path) as f:
        file = json.loads(f.read())
    return file

def client_set():
    key = load_json(API_KEY)
    client = OpenAI(
        api_key=key['api_key']
        )
    return client

def load_data(data_path,lower = 1000, upper = 4000 ,type = 'train'):
    if not os.path.exists(os.path.join(BASE_DIR, 'filter_pubmed.csv')):
        pubmed = load_dataset(data_path, trust_remote_code=True)
        pubmed = pd.DataFrame(pubmed[type])
        pubmed = pubmed.dropna()
        condition1 = pubmed['article'].map(lambda x: len(x)) > lower
        condition2 = pubmed['article'].map(lambda x: len(x)) < upper
        pubmed = pubmed[condition1&condition2]
        pubmed.to_csv('filter_pubmed.csv', index = False)
        return pubmed
    else:
        df = pd.read_csv('filter_pubmed.csv')
        return df

def chat_api(client, message, system, user, model_name):
    response = client.chat.completions.create(
        model= model_name,  # 또는 다른 모델을 사용
        messages= message)
    response = response.model_dump_json()
    response = json.loads(response)
    return response['choices'][0]['message']['content']

