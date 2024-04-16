import logging
import pickle
# import librosa
import json
import re
import os
# from settings import *
#from youtube2audio import downloadYouTubeVideo, convertVideo2Audio
# from text2summ import *
# from audio2text import convertAudio2Text
# from utils import videoUtils, audioUtils, domainFlow, llmUtils
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from operator import itemgetter
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import time
import pandas as pd
import torch
import numpy as np

logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)  # 0 for first GPU

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    #print(f"[{name}] done in {time.time() - t0:.3f} s")

seed = 2024
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"


def find_peak_from_svg(data,length):
    # SVG방식을 좌표로 바꾸기
    pattern = re.compile(r'C ((?:-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)? ){2}-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?)')
    for match in pattern.findall(data):
        coords = match.split()
        data = data.replace('C ' + match, 'L ' + coords[-1])
    #좌표들 전처리
    parts = data.split()
    res = []
    for part in parts:
        if part in ['L','M']:
            continue
        coords = part.split(',')
        res.append([float(coords[0]), float(coords[1])])
    #peak 즉 극소 점 찾기, 
    peaks = []
    for i in range(1, len(res)-1):
        if res[i][1] < res[i - 1][1] and res[i][1] < res[i + 1][1]:
            peaks.append(res[i])
    return [[int(length*x/1000),y] for x,y in peaks if y < 75]

def cos_sim(A,B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None):
    peaks = []  # peak의 인덱스를 저장할 리스트
    properties = {}  # peak의 특성을 저장할 딕셔너리
    # peak를 찾는 로직
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
    # 각 조건에 따라 peak를 선택하고 특성을 계산
    if height is not None:
        peaks = [p for p in peaks if x[p] >= height]
    if threshold is not None:
        peaks = [p for p in peaks if x[p] - x[p-1] >= threshold and x[p] - x[p+1] >= threshold]
    if distance is not None:
        peaks = [peaks[i] for i in range(len(peaks)) if i == 0 or peaks[i] - peaks[i-1] >= distance]
    if prominence is not None:
        pass  # prominence 조건 처리
    if width is not None:
        pass  # width 조건 처리
    # 특성 계산 및 반환
    properties['peak_heights'] = [x[p] for p in peaks]
    properties['left_edges'] = [p-1 for p in peaks]
    properties['right_edges'] = [p+1 for p in peaks]
    return peaks, properties
    
def preprocess_summ(summary):
    pattern = r'"\d+":\s"([^"]+)"'  # 패턴 수정
    # Find all matches using the pattern            
    matches = re.findall(pattern, summary)
    if len(matches) == 0:
        pattern = r'\d+:\s"([^"]+)"'
        matches = re.findall(pattern, summary)
        if len(matches) == 0:
            print(summary)
    top_5 = matches[:5]
    summary = top_5
    return summary



# def get_score(results, peak):
#     points = 0
#     for y,_ in peak:
#         for start,end in results:
#             if start<=y<=end:
#                 points+=1
#                 break   
#     score = min(points/max_score,1)
#     return score
def get_score(results, peak):
    score = 0
    for y,_ in peak:
        for start,end in results:
            if start<=y<=end:
                score=1
                break   
    return score

if __name__ == '__main__':
    cache_dir = './models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('final_data_large_real_final.json' , 'r') as f:
        datasets = json.load(f)

    text_model_list = ['TFIDF', 'BM25', 'DPR', 'SB', 'CB']
    topic_list = ['Comedy', 'Travel', 'Human&Blog', 'Science', 'Knowhow&Style', 'Education', 'Animals', 'Entertainment', 'News&Politics']

    score_type = "vision"
    print(score_type)
    df_result = pd.DataFrame({"category": topic_list})


#summary_1,3,5,10 구하기(각자 수정해야할 부분)

    for text_model in text_model_list:
        # 데이터를 열 기준으로 필요한 것만 선택(자기 모델의 results만)
        text_results_list = []
        clip_results_list = []
        clip4clip_results_list = []
        clip_all_results_2_list = []
        clip4clip_all_results_2_list = []
        clip_all_results_4_list = []
        clip4clip_all_results_4_list = []
        clip_all_results_5_list = []
        clip4clip_all_results_5_list = []
        clip_all_results_6_list = []
        clip4clip_all_results_6_list = []
        clip_all_results_8_list = []
        clip4clip_all_results_8_list = []
        for topic in topic_list:
            # 데이터를 행 기준으로 필요한 것만 선택(자기 category results만)
            data = [[datasets[x],x] for x in datasets.keys() if datasets[x]['category'] == topic]
            non_peak = 0
            text_results_score = 0
            clip_results_score = 0
            clip4clip_results_score = 0
            clip_all_results_2_score = 0
            clip4clip_all_results_2_score = 0
            clip_all_results_4_score = 0
            clip4clip_all_results_4_score = 0
            clip_all_results_5_score = 0
            clip4clip_all_results_5_score = 0
            clip_all_results_6_score = 0
            clip4clip_all_results_6_score = 0
            clip_all_results_8_score = 0
            clip4clip_all_results_8_score = 0
            for info,video_id in data:
                #print(video_id)
                #results가 있는 항목만 선택 -> #필터링(results가 없는게 있으면 제외)
                if score_type == 'text':
                    # if len(list(datasets[video_id].keys())) != 24:
                    #     non_peak+=1
                    title,category,channel,duration,svg,timestamps,script_time,summary_1,summary_3,summary_5,summary_10 = list(info.values())[:11]
                elif score_type == 'vision':
                    # if len(list(datasets[video_id].keys())) != 82:
                    #     non_peak+=1
                    #     continue
                    title,category,channel,duration,svg,timestamps,script_time,summary_1,summary_3,summary_5,summary_10,vision_tensor,vision_interval,clip4clip_score,clip4clip_all_score = list(info.values())[:15]
                    sum_idx = [[0,1],[1,4],[4,9],[9,19]]
                    sum_index = 0#수정 부분# summ1,3,5,10+base/large
                    sum_idx_s,sum_idx_e = [sum_idx[sum_index][0],sum_idx[sum_index][1]] 
                    text_interval,clip_interval,clip4clip_interval,clip_all_interval_2,clip4clip_all_interval_2,clip_all_interval_4,clip4clip_all_interval_4,clip_all_interval_5,clip4clip_all_interval_5,clip_all_interval_6,clip4clip_all_interval_6,clip_all_interval_8,clip4clip_all_interval_8 = datasets[video_id][f'{text_model}'][sum_idx_s:sum_idx_e],datasets[video_id]['clip_only'][sum_idx_s:sum_idx_e],datasets[video_id]['clip4clip_only'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip_2'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip4clip_2'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip_4'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip4clip_4'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip_5'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip4clip_5'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip_6'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip4clip_6'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip_8'][sum_idx_s:sum_idx_e],datasets[video_id][f'{text_model}_text_clip4clip_8'][sum_idx_s:sum_idx_e]

                peak = find_peak_from_svg(svg,duration)
                peak.sort(key = lambda x : x[1])
                peak = peak[:1] # top1만
                max_score = min(len(peak),len(clip_interval[sum_idx_s:sum_idx_e]))
                if max_score==0:
                    non_peak += 1
                    continue

                score = get_score(text_interval,peak)
                text_results_score+=score
                
                # Vision 점수 추가
                if score_type == 'vision':
                    score = get_score(clip_interval,peak)
                    clip_results_score+=score
                    score = get_score(clip4clip_interval,peak)
                    clip4clip_results_score+=score
     
                    # Text + Vision
                    score = get_score(clip_all_interval_2,peak)
                    clip_all_results_2_score+=score
                    score = get_score(clip4clip_all_interval_2,peak)
                    clip4clip_all_results_2_score+=score
                    score = get_score(clip_all_interval_4,peak)
                    clip_all_results_4_score+=score
                    score = get_score(clip4clip_all_interval_4,peak)
                    clip4clip_all_results_4_score+=score
                    score = get_score(clip_all_interval_5,peak)
                    clip_all_results_5_score+=score
                    score = get_score(clip4clip_all_interval_5,peak)
                    clip4clip_all_results_5_score+=score
                    score = get_score(clip_all_interval_6,peak)
                    clip_all_results_6_score+=score
                    score = get_score(clip4clip_all_interval_6,peak)
                    clip4clip_all_results_6_score+=score
                    score = get_score(clip_all_interval_8,peak)
                    clip_all_results_8_score+=score
                    score = get_score(clip4clip_all_interval_8,peak)
                    clip4clip_all_results_8_score+=score      
                #print('text_results_score2 :', text_results_score)
            text_results_list.append(text_results_score/(len(data)-non_peak))
            #print((len(data)-non_peak))
            #print(text_results_score)48/90
            if score_type == 'vision':
                clip_results_list.append(clip_results_score/(len(data)-non_peak))
                clip4clip_results_list.append(clip4clip_results_score/(len(data)-non_peak))

                clip_all_results_2_list.append(clip_all_results_2_score/(len(data)-non_peak))
                clip4clip_all_results_2_list.append(clip4clip_all_results_2_score/(len(data)-non_peak))
                clip_all_results_4_list.append(clip_all_results_4_score/(len(data)-non_peak))
                clip4clip_all_results_4_list.append(clip4clip_all_results_4_score/(len(data)-non_peak))
                clip_all_results_5_list.append(clip_all_results_5_score/(len(data)-non_peak))
                clip4clip_all_results_5_list.append(clip4clip_all_results_5_score/(len(data)-non_peak))
                clip_all_results_6_list.append(clip_all_results_6_score/(len(data)-non_peak))
                clip4clip_all_results_6_list.append(clip4clip_all_results_6_score/(len(data)-non_peak))
                clip_all_results_8_list.append(clip_all_results_8_score/(len(data)-non_peak))
                clip4clip_all_results_8_list.append(clip4clip_all_results_8_score/(len(data)-non_peak))

            # print(clip4clip_all_results_8_score,len(data)-non_peak)
            # exit()
        df_result[f'{text_model}_text_only'] = text_results_list
        if score_type == 'vision':
            df_result['clip_only'] = clip_results_list
            df_result['clip4clip_only'] = clip4clip_results_list

            df_result[f'{text_model}+clip_2'] = clip_all_results_2_list
            df_result[f'{text_model}+clip4clip_2'] = clip4clip_all_results_2_list
            df_result[f'{text_model}+clip_4'] = clip_all_results_4_list
            df_result[f'{text_model}+clip4clip_4'] = clip4clip_all_results_4_list
            df_result[f'{text_model}+clip_5'] = clip_all_results_5_list
            df_result[f'{text_model}+clip4clip_5'] = clip4clip_all_results_5_list
            df_result[f'{text_model}+clip_6'] = clip_all_results_6_list
            df_result[f'{text_model}+clip4clip_6'] = clip4clip_all_results_6_list
            df_result[f'{text_model}+clip_8'] = clip_all_results_8_list
            df_result[f'{text_model}+clip4clip_8'] = clip4clip_all_results_8_list

        #df_result.to_csv('final_interval.csv', index=False)
    df_clip = df_result['clip_only']
    df_clip4clip = df_result['clip4clip_only']

    df_result = df_result.drop('clip_only', axis=1)
    df_result = df_result.drop('clip4clip_only', axis=1)
    df_result = pd.concat([df_result,df_clip],axis=1)
    df_result = pd.concat([df_result,df_clip4clip],axis=1)
    df_result.to_csv(f'final_sum{sum_index}_large_score.csv', index=False)