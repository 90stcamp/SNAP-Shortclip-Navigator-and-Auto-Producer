from rouge import Rouge
import pandas as pd
from settings import *
import numpy as np

def get_Rouge_score(list_generated,list_abstract):
    rouge = Rouge()
    dict_score=rouge.get_scores(list_generated,list_abstract,avg=True)
    return dict_score

def get_rouge_from_df(generate_df, rouge_type = 'rouge-l', metric = 'f'):
    df = pd.read_csv(os.path.join(OUT_DIR, generate_df))   
    value = 0
    for idx, row in df.iterrows():
        value_dic = get_Rouge_score(row['generate'], row['abstract'])
        value += value_dic[rouge_type][metric]
    return value / len(df)

def get_rouge_list_from_all_df(save_name):
    file_list = [i for i in os.listdir(OUT_DIR) if save_name in i]
    value_list = []
    for file in file_list:
        value = get_rouge_from_df(file)
        value_list.append(value)
    value_list = np.array(value_list)
    return value_list        

def statistic_from_rouge_list(rouge_list):
    mean = np.mean(rouge_list)
    std = np.std(rouge_list)
    print(f"Mean :{mean}")
    print(f"Standard Deviation:{std}")
    return mean, std
    