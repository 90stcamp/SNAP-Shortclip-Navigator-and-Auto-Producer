from rouge import Rouge
import pandas as pd
from settings import *
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer


def get_Rouge_score(list_generated,list_abstract):
    rouge = Rouge()
    dict_score=rouge.get_scores(list_generated,list_abstract,avg=True)
    return dict_score

def get_rouge_from_df(generate_df, rouge_type = 'rouge-l', metric = 'f'):
    df = pd.read_csv(os.path.join(OUT_DIR, generate_df))   
    value = 0
    for idx, row in df.iterrows():
        try:
            value_dic = get_Rouge_score(row['generate'], row['abstract'])
            value += value_dic[rouge_type][metric]
        except:
            continue
    return value / len(df)

def get_rouge_list_from_all_df(save_name, lower, upper, df_name):
    find_file_name =  f"{save_name}_{lower}_{upper}_{df_name}"
    file_list = [i for i in os.listdir(OUT_DIR) if '_'.join(i.split('_')[:-1]) == find_file_name]
    value_list = []
    for file in tqdm(file_list, desc = 'Get Rouge List From all Dataframe', total = len(file_list)):
        value = get_rouge_from_df(file)
        value_list.append(value)
    value_list = np.array(value_list)
    return value_list        

def statistic_from_rouge_list(result_name):
    rouge_dic = np.load(os.path.join(STATS_DIR, result_name), allow_pickle= 'TRUE').item()
    mean = round(np.mean(rouge_dic['values']), 3)
    std = round(np.std(rouge_dic['values']), 3)
    print('Rouge List: ', rouge_dic['values'])
    print(f"Mean :{mean}")
    print(f"Standard Deviation:{std}")

def save_rouge_avg(avg_array, save_name):
    result_dic = {'model_name':save_name, 'values':avg_array}
    if not os.path.exists(os.path.join(STATS_DIR, f"{save_name}_result.npy")):
        np.save(os.path.join(STATS_DIR, f"{save_name}_result.npy"), result_dic)
    else:
        print('기존 파일이 존재합니다. 다른 save_name을 설정해주세요.')
        raise

def top_k_text(text, shorts_time =60,k = 3 ,interval = 1):
    candidates = [] #쇼츠 후보군들
    # shorts_time = 60 #쇼츠후보들의 초수
    # k = 3            #몇개의 후보를 선정할건지
    # interval = 1     #몇개의 문장을 기준으로 나눌껀지

    for i in range(0,len(text[0]),interval):
        start = text[0][i][1]       #시작시간
        temptext = text[0][i][0]    #첫문장
        for j in range(i+1,len(text[0])):
            end = text[0][j][2]     #끝나는시간
            temptext= ' '.join([temptext,text[0][j][0]])
            if end-start>shorts_time:
                break               #계속 합치다가 60초가 넘어가면 break
        candidates.append([temptext,start,end]) #60초동안의 문장,시작시간,끝나는시간 저장
        if j == len(text[0])-1:     #끝나는시간이 영상끝이라면 끝
            break
        
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    for i, candidate in enumerate(candidates):
        scores = scorer.score(text[1], candidate[0])    #우리가 요약한 문장과 쇼츠 후보와 rouge1점수 비교
        candidates[i] = [*candidate,scores['rouge1'][2]]#[60초동안의 문장, 시작시간,끝나는시간, rouge1_F1점수]

    candidates.sort(reverse = True, key = lambda x : x[3])#rouge1_F1점수로 sort

    final_candidates = candidates[:k]
    return final_candidates

