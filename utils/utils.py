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

def load_data(path, type = 'train'):
    pubmed = load_dataset(path, trust_remote_code=True)
    pubmed_train = pd.DataFrame(pubmed[type])
    return pubmed_train

def chat_api(client, message, system, user, model_name):
    response = client.chat.completions.create(
        model= model_name,  # 또는 다른 모델을 사용
        messages= message)
    response = response.model_dump_json()
    response = json.loads(response)
    return response['choices'][0]['message']['content']

