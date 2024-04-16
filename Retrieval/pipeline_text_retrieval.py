import logging
import pickle
# import librosa
import json
import re
import os
from settings import *
from youtube2audio import downloadYouTubeVideo, convertVideo2Audio
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


# 단일 text 
def make_candidates_text(timestamps,shorts_time=60):
    candidates = []
    for i in range(len(timestamps)):
        current_start, current_end = timestamps[i]['timestamp']
        temptext = timestamps[i]['text']
        merged_data = [current_start, current_end]
        for j in range(i+1, len(timestamps)):
            next_end = timestamps[j]['timestamp'][1]
            if next_end - current_start <= shorts_time:
                merged_data[1] = next_end
                temptext += ' ' + timestamps[j]['text']
            else:
                break
        candidates.append([temptext, merged_data[0], merged_data[1]])
    return candidates

# vision+text 
def make_candidates_vision(timestamps,section):
    candidates = [0]*len(section)
    for i,(s,e) in enumerate(section):
        candidates[i] = {'text' : '',"timestamp" : [[s],[e]]}
        for j in range(len(timestamps)):
            sts,text = timestamps[j]['timestamp'],timestamps[j]['text']
            if sts[0]<=s<=sts[1]<=e or s<=sts[0]<=sts[1]<=e or s<=sts[0]<=e<=sts[1]:
                #겹치는 구간이 있으면
                candidates[i]['text']+=text
                candidates[i]['timestamp'][0].append(sts[0])
                candidates[i]['timestamp'][1].append(sts[1])
        candidates[i]['timestamp'][0] = min(candidates[i]['timestamp'][0])
        candidates[i]['timestamp'][1] = max(candidates[i]['timestamp'][1])
        if (candidates[i]['timestamp'][1] - candidates[i]['timestamp'][0]) > 60:
            candidates[i]['timestamp'][1] = candidates[i]['timestamp'][0]+60
        candidates[i] = [candidates[i]['text'],*candidates[i]['timestamp']]
    return candidates

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
    
def preprocess_summ(summary, k):
    pattern = r'"\d+":\s"([^"]+)"'  # 패턴 수정
    # Find all matches using the pattern            
    matches = re.findall(pattern, summary)
    if len(matches) == 0:
        pattern = r'\d+:\s"([^"]+)"'
        matches = re.findall(pattern, summary)
        if len(matches) == 0:
            pattern = r'"\d+".\s"([^"]+)"'  # 패턴 수정
            matches = re.findall(pattern, summary)
            if len(matches) == 0:
                pattern = r'\d+.\s"([^"]+)"'
                matches = re.findall(pattern, summary)
                if len(matches) == 0:
                    pattern = r'\d+:\s([^"]+)"'
                    matches = re.findall(pattern, summary)
                    if len(matches) == 0:
                        pattern = r'\d+.\s([^"]+)"'
                        matches = re.findall(pattern, summary)
                        if len(matches) == 0:
                            pattern = r'\d+:\s*(.*)'
                            matches = re.findall(pattern, summary)
                            if len(matches) == 0:
                                pattern = r'\d+.\s*(.*)'
                                matches = re.findall(pattern, summary)
                                if len(matches) == 0:
                                    print(summary)
            
    top_k = matches[:k]
    summary = top_k
    return summary

def tfidf_score(script, summary):
    Vectorizer = TfidfVectorizer().fit(script)
    corpusVec = Vectorizer.transform(script)
    tfidf_score = np.empty((0,len(script)))
    for r_origin_topic in summary:
        r_origin_vec = Vectorizer.transform([r_origin_topic])
        value = np.dot(corpusVec,r_origin_vec.T).toarray().flatten()
        tfidf_score = np.append(tfidf_score,np.array([value]),axis=0)
    text_score = tfidf_score.T
    return text_score

def bm25_score(script, summary):
    tokenized_script = [word_tokenize(doc) for doc in script]
    bm25 = BM25Okapi(tokenized_script)
    bm25_score = np.empty((0, len(script)))
    for r_origin_topic in summary:
        r_origin_vec = word_tokenize(r_origin_topic)
        scores_ = bm25.get_scores(r_origin_vec)
        bm25_score = np.append(bm25_score, np.array([scores_]), axis=0)               
    text_score = bm25_score.T
    return text_score

def dpr_score(script, summary):
    input_ids = tokenizer(summary, return_tensors="pt", padding=True, max_length=512, truncation=True)["input_ids"].to(device)
    embeddings_summ_dpr = text_model_dpr(input_ids).pooler_output.detach().cpu()
    batch_size = 32
    text_score = torch.empty(0).to('cpu')
    for i in range(0, len(script), batch_size):
        torch.cuda.empty_cache()
        batch = script[i:i + batch_size]
        input_ids = tokenizer(batch, return_tensors="pt", padding=True, max_length=512, truncation=True)["input_ids"].to(device)
        embeddings_text_dpr = text_model_dpr(input_ids).pooler_output.detach().cpu()
        sim_text_dpr = cos_sim(embeddings_text_dpr,embeddings_summ_dpr.T)
        text_score = torch.cat((text_score, torch.tensor(sim_text_dpr)), dim=0)
    return text_score

def sb_score(script, summary):
    embeddings_summ_sb = text_model_sb.encode(summary)
    embeddings_text_sb = text_model_sb.encode(script)
    text_score = cos_sim(embeddings_text_sb, embeddings_summ_sb.T)
    return text_score

def cb_score(script, summary):
    embeddings_summ_cb = text_model_cb.encode(summary)
    embeddings_text_cb = text_model_cb.encode(script)
    text_score = cos_sim(embeddings_text_cb, embeddings_summ_cb.T)
    return text_score

def scaling(score):
    scaler = RobustScaler()
    scaler.fit(score)
    score = scaler.transform(score).T
    return score

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
    text_model_dpr = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", cache_dir=cache_dir).to(device)
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", cache_dir=cache_dir)
    text_model_sb = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', cache_folder=cache_dir).to(device)
    text_model_cb = SentenceTransformer('colbert-ir/colbertv2.0', cache_folder=cache_dir).to(device)

    with open('final_data_vision_clip_base_final.json' , 'r') as f:
        datasets = json.load(f)
    topic_list = ['Comedy', 'Travel', 'Human&Blog', 'Science', 'Knowhow&Style', 'Education', 'Animals', 'Entertainment', 'News&Politics']
    text_model_list = ['TFIDF', 'BM25', 'DPR', 'SB', 'CB']
    score_type = "vision"
    print(score_type)
    df_result = pd.DataFrame({"category": topic_list})

    for text_model in text_model_list:
        for video_id in datasets.keys():
            print(video_id)
            if score_type == 'text':
                # if len(datasets[video_id].keys())!=11:
                #     continue
                title,category,channel,duration,svg,timestamps,script_time,summary_1,summary_3,summary_5,summary_10 = list(datasets[video_id].values())[:11]
            elif score_type == 'vision':
                # if len(datasets[video_id].keys())!=15:
                #     continue
                title,category,channel,duration,svg,timestamps,script_time,summary_1,summary_3,summary_5,summary_10,vision_tensor,vision_interval,clip4clip_score,clip4clip_all_score = list(datasets[video_id].values())[:15]


            # 수정부분1
            if category not in ['Human&Blog', 'Education']:
                continue

            summary_1 = preprocess_summ(summary_1, 1)
            summary_3 = preprocess_summ(summary_3, 3)
            summary_5 = preprocess_summ(summary_5, 5)
            summary_10 = preprocess_summ(summary_10, 10)
            summary = sum([summary_1,summary_3,summary_5,summary_10], []) #[:1, 1:4, 4:9, 9:19]

            # get script
            timecand = make_candidates_text(timestamps)
            script = []
            for idx in timecand:
                script.append(idx[0])

            # 실험용 코드
            timecand = [[t,s,s+59] for t,s,e in timecand]
            #print(np.mean([(z-y) for x,y,z in timecand]))

            # get text score
            torch.cuda.empty_cache()
            if text_model=='TFIDF':
                text_score = tfidf_score(script, summary)
            elif text_model=='BM25':
                text_score = bm25_score(script, summary)
            elif text_model=='DPR':
                text_score = dpr_score(script, summary)
            elif text_model=='SB':
                text_score = sb_score(script, summary)
            elif text_model=='CB':
                text_score = cb_score(script, summary)
            text_score_only = text_score
            text_score = scaling(text_score)
            text_results = []
            for sen in range(text_score.shape[0]):
                piv = np.argmax(text_score[sen])
                text_results.append(timecand[piv][1:])

            # Vision 점수 추가
            if score_type == 'vision':
                # 실험용 코드
                timecand = [[s,s+59] for s,e in vision_interval]
                #print(np.mean([(y-x) for x,y in timecand]))

                # vision only
                vision_tensor = torch.tensor(vision_tensor)
                try:
                    scene_score = [vision_tensor[x[0]:x[1]] for x in timecand]
                    scene_score_class = torch.stack([x.mean(dim=0) for x in scene_score])
                    scene_score_class = torch.where(torch.isnan(scene_score_class), torch.zeros_like(scene_score_class), scene_score_class) #nan 0으로
                except:
                    continue

                clip_score = scaling(scene_score_class)
                clip_results = []
                for sen in range(clip_score.shape[0]):
                    piv = np.argmax(clip_score[sen])
                    clip_results.append(timecand[piv])
                
                
                clip4clip_score = scaling(clip4clip_score)
                clip4clip_results = []
                for sen in range(clip4clip_score.shape[0]):
                    piv = np.argmax(clip4clip_score[sen])
                    clip4clip_results.append(timecand[piv])


                # Text + Vision
                timecand = make_candidates_vision(timestamps,vision_interval)


                # 실험용 코드
                timecand = [[t,s,s+59] for t,s,e in timecand]
                #print(np.mean([(z-y) for x,y,z in timecand]))
                script = []
                for idx in timecand:
                    script.append(idx[0])
                # timecand가 바뀌기 때문에 다시 해야 함
                # get text score
                torch.cuda.empty_cache()
                if text_model=='TFIDF':
                    text_score = tfidf_score(script, summary)
                elif text_model=='BM25':
                    text_score = bm25_score(script, summary)
                elif text_model=='DPR':
                    text_score = dpr_score(script, summary)
                elif text_model=='SB':
                    text_score = sb_score(script, summary)
                elif text_model=='CB':
                    text_score = cb_score(script, summary)
                text_score_all = text_score
                
                scene_score = [vision_tensor[int(x[1]):int(x[2])] for x in timecand]
                scene_score_class = torch.stack([x.mean(dim=0) for x in scene_score])
                scene_score_class = torch.where(torch.isnan(scene_score_class), torch.zeros_like(scene_score_class), scene_score_class) #nan 0으로
                clip_score = scaling(scene_score_class)

                
                # Text 점수 합산
                text_score = scaling(text_score)
                clip4clip_all_score = scaling(clip4clip_all_score)
                clip_all_score_2 = (clip_score*0.2) + (text_score*0.8)
                clip_all_results_2 = []
                for sen in range(clip_all_score_2.shape[0]):
                    piv = np.argmax(clip_all_score_2[sen])
                    clip_all_results_2.append(timecand[piv][1:])
                clip4clip_all_score_2 = (clip4clip_all_score*0.2) + (text_score*0.8)
                clip4clip_all_results_2 = []
                for sen in range(clip4clip_all_score_2.shape[0]):
                    piv = np.argmax(clip4clip_all_score_2[sen])
                    clip4clip_all_results_2.append(timecand[piv][1:])

                clip_all_score_4 = (clip_score*0.4) + (text_score*0.6)
                clip_all_results_4 = []
                for sen in range(clip_all_score_4.shape[0]):
                    piv = np.argmax(clip_all_score_4[sen])
                    clip_all_results_4.append(timecand[piv][1:])
                clip4clip_all_score_4 = (clip4clip_all_score*0.4) + (text_score*0.6)
                clip4clip_all_results_4 = []
                for sen in range(clip4clip_all_score_4.shape[0]):
                    piv = np.argmax(clip4clip_all_score_4[sen])
                    clip4clip_all_results_4.append(timecand[piv][1:])

                clip_all_score_5 = (clip_score*0.5) + (text_score*0.5)
                clip_all_results_5 = []
                for sen in range(clip_all_score_5.shape[0]):
                    piv = np.argmax(clip_all_score_5[sen])
                    clip_all_results_5.append(timecand[piv][1:])
                clip4clip_all_score_5 = (clip4clip_all_score*0.5) + (text_score*0.5)
                clip4clip_all_results_5 = []
                for sen in range(clip4clip_all_score_5.shape[0]):
                    piv = np.argmax(clip4clip_all_score_5[sen])
                    clip4clip_all_results_5.append(timecand[piv][1:])

                clip_all_score_6 = (clip_score*0.6) + (text_score*0.4)
                clip_all_results_6 = []
                for sen in range(clip_all_score_6.shape[0]):
                    piv = np.argmax(clip_all_score_6[sen])
                    clip_all_results_6.append(timecand[piv][1:])
                clip4clip_all_score_6 = (clip4clip_all_score*0.6) + (text_score*0.4)
                clip4clip_all_results_6 = []
                for sen in range(clip4clip_all_score_6.shape[0]):
                    piv = np.argmax(clip4clip_all_score_6[sen])
                    clip4clip_all_results_6.append(timecand[piv][1:])

                clip_all_score_8 = (clip_score*0.8) + (text_score*0.2)
                clip_all_results_8 = []
                for sen in range(clip_all_score_8.shape[0]):
                    piv = np.argmax(clip_all_score_8[sen])
                    clip_all_results_8.append(timecand[piv][1:])
                clip4clip_all_score_8 = (clip4clip_all_score*0.8) + (text_score*0.2)
                clip4clip_all_results_8 = []
                for sen in range(clip4clip_all_score_8.shape[0]):
                    piv = np.argmax(clip4clip_all_score_8[sen])
                    clip4clip_all_results_8.append(timecand[piv][1:])



            if score_type == 'vision':
                datasets[video_id]['clip_only'] = clip_results
                datasets[video_id]['clip4clip_only'] = clip4clip_results
            datasets[video_id][text_model] = text_results
            if score_type == 'vision':
                datasets[video_id][f'{text_model}_text_clip_2'] = clip_all_results_2
                datasets[video_id][f'{text_model}_text_clip4clip_2'] = clip4clip_all_results_2
                datasets[video_id][f'{text_model}_text_clip_4'] = clip_all_results_4
                datasets[video_id][f'{text_model}_text_clip4clip_4'] = clip4clip_all_results_4
                datasets[video_id][f'{text_model}_text_clip_5'] = clip_all_results_5
                datasets[video_id][f'{text_model}_text_clip4clip_5'] = clip4clip_all_results_5
                datasets[video_id][f'{text_model}_text_clip_6'] = clip_all_results_6
                datasets[video_id][f'{text_model}_text_clip4clip_6'] = clip4clip_all_results_6
                datasets[video_id][f'{text_model}_text_clip_8'] = clip_all_results_8
                datasets[video_id][f'{text_model}_text_clip4clip_8'] = clip4clip_all_results_8

            datasets[video_id][f'{text_model}_text_score_only'] = text_score_only.tolist()
            datasets[video_id][f'{text_model}_text_score_all'] = text_score_all.tolist()

        with open(f'final_data_base_1.json', 'w', encoding='utf-8') as file:
            json.dump(datasets, file, ensure_ascii=False, indent=4)

            # with open('final_data_base_score_version.json', 'w', encoding='utf-8') as file:
            #     json.dump(datasets_score, file, ensure_ascii=False, indent=4)