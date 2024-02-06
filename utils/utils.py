from settings import *
import numpy as np 
import matplotlib.pyplot as plt 
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
        pubmed = load_dataset(data_path)
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



def plot_scatter2D(dataframe, sampling=10_000, save=False, fname='scatter_plot'):
    color_map = ['#8B0000', '#FF0000', '#BDB76B', '#7CFC00', '#008080', '#4169E1', '#FF69B4']
    if sampling:
        idx = np.random.randint(0, len(dataframe), sampling)    
        samples = dataframe.iloc[idx]
    else:
        samples = dataframe.copy()
    
    x = samples.iloc[:, 0]
    y = samples.iloc[:, 1]
    label = samples.loc[:, 'label']
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.scatter(x, y, color=label.apply(lambda x: color_map[x]), alpha=0.5)
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=200)
    plt.show()
    
def plot_scatter3D(dataframe, sampling=10_000, save=False, fname='scatter_plot'):
    color_map = ['#8B0000', '#FF0000', '#BDB76B', '#7CFC00', '#008080', '#4169E1', '#FF69B4']
    if sampling:
        idx = np.random.randint(0, len(dataframe), sampling)    
        samples = dataframe.iloc[idx]
    else:
        samples = dataframe.copy()
    
    x = samples.iloc[:, 0]
    y = samples.iloc[:, 1]
    z = samples.iloc[:, 2]
    label = samples.loc[:, 'label']
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=label.apply(lambda x: color_map[x]), alpha=0.5)
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'{fname}.png'), dpi=200)
    plt.show()
    
