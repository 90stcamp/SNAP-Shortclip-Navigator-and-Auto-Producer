import logging
import pickle
import json
import re
import os
from contextlib import contextmanager
from operator import itemgetter
import numpy as np
import torch
import time
from pytube import YouTube
import cv2
from natsort import natsorted
import glob
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, CLIPTextModel
import pandas as pd
import shutil
from scenedetect import detect, ContentDetector, AdaptiveDetector, split_video_ffmpeg
import subprocess
from youtube2audio import downloadYouTubeVideo

logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

######os.environ["TOKENIZERS_PARALLELISM"] = "false"
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)  # 0 for first GPU

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



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    #print(f"[{name}] done in {time.time() - t0:.3f} s")

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

def downloadYouTubeVideo(video_id, videourl):
    command = ['yt-dlp', '-f', 'bestvideo[height<=720][ext=mp4]', '-o', f'videos/{video_id}.mp4', videourl]
    subprocess.run(command)

def get_video_info(filepath):
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)

    #불러온 비디오 파일의 정보 출력
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)
    return video, fps, length

def save_video_frame(filepath, video, fps, per_min):
    #프레임을 저장할 디렉토리를 생성
    try:
        if not os.path.exists(filepath[:-4]):
            os.makedirs(filepath[:-4])
    except OSError:
        print ('Error: Creating directory. ' +  filepath[:-4])

    print('Splitting frame start...')
    ret = True
    while(ret):
        ret, image = video.read()
        if(int(video.get(1) % fps) == per_min): #앞서 불러온 fps 값을 사용하여 1초마다 추출
            cv2.imwrite(filepath[:-4] + "/frame%d.jpg" % int(video.get(1)//fps), image)
            #print('Saved frame number :', str(int(video.get(1)//fps)))
    print('Splitting frame done :', str(int(video.get(1)//fps)))
    video.release()
    cv2.destroyAllWindows()

# #기존
# def get_visual_scores(model, processor, device, summarization, image_list, batch_size):
#     # class_ = text.split('\n') # 다른 모듈과 일치 시키기 위해 바꿔줘야함
#     # class_name = class_[1:-1]
#     images = [Image.open(image) for image in image_list]
#     # 빈 텐서 생성
#     score_tensor = torch.empty(0).to(device)

#     print('Measuring Similarity...')
#     for i in tqdm(range(0, len(images), batch_size)):
#         batch = images[i:i + batch_size]
#         inputs = processor(text=summarization, images=batch, return_tensors="pt", padding=True, truncation=True)

#         outputs = model(**inputs.to(device))

#         logits_per_image = outputs.logits_per_image # this is the image-text similarity score
#         probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
#         probs = logits_per_image
#         print(probs)
#         # 스코어를 텐서에 추가
#         score_tensor = torch.cat((score_tensor, probs.detach()), dim=0)
#     print('Done!')
#     return score_tensor

def get_visual_scores(text_model, text_processor, vision_model, vision_processor, model, processor, device, summarization, image_list, batch_size, timecand_all, clip_interval):
    timecand_all = [x[1:] for x in timecand_all]
    # print(np.mean([(x-y) for x,y in timecand_all]))
    # print(np.mean([(x-y) for x,y in clip_interval]))
    images = [Image.open(image) for image in image_list]

    print('Measuring Similarity...')
    text_inputs = text_processor(text=summarization, return_tensors="pt", padding=True, truncation=True)
    text_emb = text_model(**text_inputs.to(device))['pooler_output'].detach().cpu()
    # 빈 텐서 생성
    clip4clip_score = torch.empty(0).to(device).cpu()
    score_tensor = torch.empty(0).to(device)
    for i in tqdm(range(0, len(images), batch_size)):
        torch.cuda.empty_cache()
        batch = images[i:i + batch_size]
        vision_inputs = vision_processor(images=batch, return_tensors="pt")
        vision_emb = vision_model.get_image_features(**vision_inputs.to(device))
        clip4clip_score = torch.cat((clip4clip_score, vision_emb.detach().cpu()), dim=0)
        torch.cuda.empty_cache()
        #clip
        inputs = processor(text=summarization, images=batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs.to(device))
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        probs = logits_per_image
        # 스코어를 텐서에 추가
        score_tensor = torch.cat((score_tensor, probs.detach()), dim=0)

        
    vision_score = []
    for interval in clip_interval:
        vision_score.append(cos_sim(text_emb, torch.mean(clip4clip_score[interval[0]:interval[1]], dim=0).T))
    vision_score = torch.tensor(vision_score)

    all_score = []
    for interval in timecand_all:
        all_score.append(cos_sim(text_emb, torch.mean(clip4clip_score[int(interval[0]):int(interval[1])], dim=0).T))
    all_score = torch.tensor(all_score)

    print('Done!')
    return vision_score, all_score, score_tensor



def split_frame(filepath, class_name, fps, length): # AdaptiveDetector, ContentDetector 중 선택 #score_tensor
    # AdaptiveDetector
    #scene_list = detect(filepath, AdaptiveDetector())

    # ContentDetector
    threshold = 35
    min_scene_len= fps*5 # fps * 시간
    scene_list = detect(filepath, ContentDetector(min_scene_len=min_scene_len, threshold=threshold))
    
    scene_time = [[int(x[0].get_frames()/fps), int(x[1].get_frames()/fps)] for x in scene_list]

    # split_video_ffmpeg(filepath, scene_list, video_name=f"videos/result/{filepath.split('/')[-1][:-4]}-{sentence+1}")

    time_60 = []
    for i in range(len(scene_time)):
        current_start = scene_time[i][0]
        current_end = scene_time[i][1]
        merged_data = [current_start, current_end]

        if (int(length/fps) - current_start) <= 60:
            break

        for j in range(i+1, len(scene_time)):
            next_end = scene_time[j][1]
            if next_end - current_start <= 60:
                merged_data[1] = next_end
            else:
                break
        time_60.append(merged_data)
    scene_time=time_60


    # scene_score = [score_tensor[x[0]:x[1]] for x in scene_time]
    # scene_score_class = torch.stack([x.mean(dim=0) for x in scene_score])
    
    # scene_score_class = torch.where(torch.isnan(scene_score_class), torch.zeros_like(scene_score_class), scene_score_class) #nan 0으로


    print(f'Splitting video by {len(scene_list)}... it takes time')
    if not os.path.exists('videos/result/'):
        os.makedirs('videos/result/')
    # k = 3 #실제는 top1만 입력
    # final_interval = []
    # for sentence in tqdm(range(score_tensor.shape[1])):
    #     top_k_value, top_k_index = torch.topk(scene_score_class[:, sentence], k, largest=True)
    #     print(f"Class: {class_name[sentence]}:")
    #     print([(itemgetter(x)(scene_time),float(y)) for x,y in zip(top_k_index,top_k_value)])
    #     final_interval.append([itemgetter(x)(scene_time) for x in top_k_index][0]) #TOP1이라 이렇게 설정
    #     #split_video_ffmpeg(filepath, itemgetter(*top_k_index)(scene_list), video_name=f"videos/result/{filepath.split('/')[-1][:-4]}-{sentence+1}")
    # for start,end in final_interval:
    #     start = f"{start//60//60}:{(start//60)%60}:{start%60}"
    #     end= f"{end//60//60}:{(end//60)%60}:{end%60}"
    #     cmd = f"ffmpeg -i /home/lgh/etc/videos/KOEfDvr4DcQ.mp4 -ss {start} -to {end} -vcodec copy -acodec copy {start}-{end}output.mp4"
    #     subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return scene_time

def transform_length(heatmap, length, fps):
    path_data = parse_path(heatmap)
    path_data = path_data[1:102]
    points = []
    for segment in path_data:
        for i in range(int(length/fps) + 1):
            t = i / (length/fps)
            point = segment.point(t)
            points.append((point.real, -point.imag))  # SVG y 좌표는 아래로 향하므로 -y를 사용

    # Convert points to numpy array
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 길이로 정규화
    x = range(len(x)//101)
    y = [y[i*101:i*101+101].mean() for i in range(len(y)//101)]
    return y
    
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

if __name__ == '__main__':
    with open('final_data_vision_clip_base_final_2.json' , 'r') as f: #중간에 끊기면, final_data_vision_clip_large_final로 바꿔서 실행 (단, 마지막 저장 파일 명도 바꿀 것 권장)
        data = json.load(f)

##### 파일 여러개 받아서 통합(저장 후 다시 불러오는 것 권장)

    model_name = 'openai/clip-vit-base-patch32' #'openai/clip-vit-large-patch14'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = CLIPTextModel.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}').to(device)
    text_processor = AutoTokenizer.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}')
    vision_model = CLIPModel.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}').to(device)
    vision_processor = CLIPProcessor.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}')
    model = CLIPModel.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}').to(device)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}')
    batch_size = 16
    for video_id in data.keys():
        if len(data[video_id].keys())==15:
            continue
        else:
            print(video_id)
            torch.cuda.empty_cache()
            ###title,category,channel,duration,svg,timestamps,script_time,summary = data[video_id].values()
            title,category,channel,duration,svg,timestamps,script_time,summary_1,summary_3,summary_5,summary_10 = data[video_id].values()

            if category not in ['Human&Blog', 'Education']:
                continue

            # k개
            summary_1 = preprocess_summ(summary_1, 1)
            summary_3 = preprocess_summ(summary_3, 3)
            summary_5 = preprocess_summ(summary_5, 5)
            summary_10 = preprocess_summ(summary_10, 10)
            summary = summary = sum([summary_1,summary_3,summary_5,summary_10], []) #[:1, 1:4, 4:9, 9:19]

            youtube_link = f'https://www.youtube.com/watch?v={video_id}'
            print(len(timestamps))
        
            #logging.info("Process: Summ-Text top-k candidates")

            timeselect=[]

            timecand_text=make_candidates_text(timestamps)
            # 실험용 코드
            timecand_text = [[t,s,s+59] for t,s,e in timecand_text]

                
            #df_interval = pd.DataFrame(columns=['ID', 'GroundTruth', 'clip'])
            #downloadYouTube(youtube_link, f_name=video_id)
            try:
                downloadYouTubeVideo(video_id, youtube_link)
            except:
                continue
            filepath = f'videos/{video_id}.mp4'
            try:
                video, fps, length = get_video_info(filepath)
            except:
                continue


            try:
                save_video_frame(filepath, video, fps, 1) # 마지막은 몇 초에 한번 프레임을 생성할 것인지
            except:
                continue

            image_list = natsorted(glob.glob(f'{filepath[:-4]}/*.jpg'))
            
            try:
                vision_interval = split_frame(filepath, summary, fps, length)
            except:
                continue
            clip_interval = [[s,s+59] for s,e in vision_interval]
            timecand_all = make_candidates_vision(timestamps,vision_interval)
            timecand_all = [[t,s,s+59] for t,s,e in timecand_all]


            vision_score, all_score, score_tensor = get_visual_scores(text_model, text_processor, vision_model, vision_processor, model, processor, device, summary, image_list, batch_size, timecand_all, clip_interval) #version1(기존)
                #score_tensor = get_visual_scores(model, processor, device, summary, image_list, batch_size, clip_interval)
                #clip_interval = split_frame(filepath, summary, fps, length)
#               except:
#                  continue                
            
            #score_tensor   #초당 유사도(5,길이)
            #clip_interval  #CANDIDATE
            #results        #예측결과(top5구간)
            # print(score_tensor)
            # print(clip_interval)
            # vision_score = score_tensor.cpu()
            
            # results = []
            # for num in range(vision_score.shape[0]):
            #     piv = np.argmax(vision_score[num])
            #     results.append(timecand[piv][1:])
            
            torch.cuda.empty_cache()  
            shutil.rmtree('videos/')
        
        
            score_tensor = score_tensor.cpu().numpy().tolist()
            vision_score = vision_score.cpu().numpy().tolist()
            all_score = all_score.cpu().numpy().tolist()
            #print(score_tensor)
            data[video_id]['vision_tensor'] = score_tensor

            data[video_id]['vision_interval'] = vision_interval

            data[video_id]['vision_score'] = vision_score
            data[video_id]['all_score'] = all_score
            # data[video_id]['results'] = results
            with open(f'final_data_vision_clip_base_final_2.json', 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

        # except:
        #     continue
        #print(topic,topic_score/(len(data[topic])))
        #total_score+=topic_score/(len(data[topic]))
    #print(total_score/(len(data)))