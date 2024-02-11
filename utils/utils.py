from settings import *
import numpy as np 
import matplotlib.pyplot as plt 
import json
from openai import OpenAI
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
from utils import preprocess

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
    file_name = f"{data_path.split('/')[-1]}.csv"

    if not os.path.exists(os.path.join(BASE_DIR, file_name)):
        # CNN dataset: version issue
        if file_name == 'cnn_dailymail.csv':
            df = load_dataset(data_path, '2.0.0')
        else: 
            df = load_dataset(data_path)

        # Reddit dataset: time issue
        if file_name == 'tldr-17.csv':
            df = pd.DataFrame(df[type].select_columns(['content','summary']))
        else:
            df = pd.DataFrame(df[type])
        rename_col_dic = {'article':'document','content': 'document','abstract': 'summary', 'highlights': 'summary'}
        df.columns = [rename_col_dic[i] if i in rename_col_dic else i for i in df.columns]
        df = df[['document', 'summary']]
        df = df[df['document'] > df['summary']]
        df.to_csv(file_name, index = False)
    
    df = pd.read_csv(file_name)
    df = df.dropna(axis = 0)
    condition1 = df['document'].map(lambda x: len(x)) > lower
    condition2 = df['document'].map(lambda x: len(x)) < upper

    return df[condition1 & condition2]

def dataset_word_statistic(data_name, lower, upper):
    file_name = f"{data_name.split('/')[-1]}.csv"
    print('Start Data Loading...')
    df = load_data(data_name, lower = lower, upper = upper)
    print('End Data Loading')
    tqdm.pandas(desc="Doc,Sum Words Length Calculate...")
    df_statistic = df.progress_applymap(lambda x: len(nltk.word_tokenize(x)))
    doc_words, sum_words = df_statistic.mean()

    tqdm.pandas(desc="Sum sentence length calculate...")
    sum_sen_len = df['summary'].map(lambda x: preprocess.get_sententence_num(x)).sum()/ df.shape[0]
    print(f'Dataset: {file_name}')
    print(f'Doc Words: {doc_words}')
    print(f'Sum Words: {sum_words}')
    print(f'Sum Sentence Length: {sum_sen_len}')

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
    
