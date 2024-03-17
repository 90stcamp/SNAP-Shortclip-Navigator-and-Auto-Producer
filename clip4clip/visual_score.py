#라이브러리 호출
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import pandas as pd
import glob
from natsort import natsorted
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from scenedetect import detect, ContentDetector, AdaptiveDetector, split_video_ffmpeg
from operator import itemgetter
import logging
from pytube import YouTube
import shutil
import re
import time
import dbs
import requests
from requests.exceptions import Timeout, RequestException
import json 
from settings import * 

def send_data_to_server(dict_file,server):
    response = requests.post(server, json=dict_file, timeout=10)


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
    return video, fps


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


# 이미 util에 있을 수도
def get_visual_scores(model, processor, device, text, image_list, batch_size):
    matches = re.findall(r'\d+\..*?\n', text)
    class_name = [match.split('.', 1)[1].strip() for match in matches]

    # class_ = text.split('\n') # 다른 모듈과 일치 시키기 위해 바꿔줘야함
    # class_name = class_[1:-1]
    images = [Image.open(image) for image in image_list]

    # 빈 텐서 생성
    score_tensor = torch.empty(0).to(device)

    print('Measuring Similarity...')
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i + batch_size]
        inputs = processor(text=class_name, images=batch, return_tensors="pt", padding=True)
        outputs = model(**inputs.to(device))
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        # 스코어를 텐서에 추가
        score_tensor = torch.cat((score_tensor, logits_per_image.detach()), dim=0)
    print('Done!')
    return score_tensor, class_name


# Visualization
def get_top_frame(score_tensor, image_list, filepath, top_k):    
    for sentence in range(score_tensor.shape[1]):
        top_k_value, top_k_index = torch.topk(score_tensor[:, sentence], top_k, largest=True)

        print(top_k_index)
        for i in range(top_k):
            index = top_k_index[i].item()
            probability = top_k_value[i].item()
            image_name = image_list[index].split('/')[-1]
            print(f'Image: {image_name}, Probability: {probability}')

            image = Image.open(f'{filepath[:-4]}/{image_name}')
            plt.imshow(image)
            plt.show()


def split_frame(scene_time, class_name, score_tensor, llm_result): # AdaptiveDetector, ContentDetector 중 선택
    print("LLM SUmmariztaion Score matrix\n", llm_result)
    scene_score = [score_tensor[x[0]:x[1]] for x in scene_time]
    # final 사용
    scene_score_vision = torch.stack([x.mean(dim=0) for x in scene_score])
    scene_score_vision = scene_score_vision.cpu()
    
    llm_result = torch.Tensor(llm_result).T
    llm_result = llm_result.softmax(dim = 1)

    scene_score_vision = scene_score_vision.softmax(dim = 1)

    scene_score_class = scene_score_vision + llm_result
    scene_score_class = torch.where(torch.isnan(scene_score_class), torch.zeros_like(scene_score_class), scene_score_class)
    
    scene_score_class_vision = scene_score_vision
    scene_score_class_llm = llm_result

    print(f'Splitting video by {len(scene_list)}... it takes time')
    if not os.path.exists('videos/result/'):
        os.makedirs('videos/result/')
    k = 1 #len(scene_time)

    final_interval_all = []
    final_interval_vision = []
    final_interval_llm = []
    
    for sentence in tqdm(range(score_tensor.shape[1])):
        top_k_value, top_k_index = torch.topk(scene_score_class[:, sentence], k, largest=True)
        print(f"Class: {class_name[sentence]}:")
        print([(itemgetter(x)(scene_time),float(y)) for x,y in zip(top_k_index,top_k_value)])
        
        final_interval_all.append([itemgetter(x)(scene_time) for x in top_k_index][0])

    for sentence in tqdm(range(score_tensor.shape[1])):
        top_k_value, top_k_index = torch.topk(scene_score_class_vision[:, sentence], k, largest=True)
        print(f"Class: {class_name[sentence]}:")
        print([(itemgetter(x)(scene_time),float(y)) for x,y in zip(top_k_index,top_k_value)])
        
        final_interval_vision.append([itemgetter(x)(scene_time) for x in top_k_index][0])

    for sentence in tqdm(range(score_tensor.shape[1])):
        top_k_value, top_k_index = torch.topk(scene_score_class_llm[:, sentence], k, largest=True)
        print(f"Class: {class_name[sentence]}:")
        print([(itemgetter(x)(scene_time),float(y)) for x,y in zip(top_k_index,top_k_value)])
        
        final_interval_llm.append([itemgetter(x)(scene_time) for x in top_k_index][0])
    
    return final_interval_all, final_interval_vision, final_interval_llm

# 아래 2개는 다른 util에도 있음
def downloadYouTube(videourl, f_name, path='videos'):
    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(output_path=path, filename=f'{f_name}.mp4')

def get_origin_text(text):
    matches = re.findall(r'\d+\..*?\n', text)# 값이 안 찾아짐
    responds = [match.split('.', 1)[1].strip() for match in matches]
    return responds

def load_txt_summarize(video_id):
    with open(f'videos/{video_id}.txt', 'r', encoding="UTF-8") as file:
        text = file.read()
    return text

def wait_for_file(file_path, timeout=500, check_interval=5):
    """Wait for a specific file to be created within a timeout period."""
    start_time = time.time()
    while not os.path.exists(file_path):
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"Timeout reached: {file_path} not found.")
            return False
        # print(f"Waiting for file: {file_path}. Checked after {elapsed_time:.2f}s")
        time.sleep(check_interval)
    print(f"File found: {file_path}")
    return True

if __name__=='__main__':
    conn = dbs.MYSQL_DATABASE_CONN
    with open(SERVER_DIR, "r") as env:
        dic_server = json.load(env)
    server = dic_server['server']

    youtube_link = os.getenv('YOUTUBE_LINK')
    logging.info("Process Started")
    video_id=youtube_link.split('watch?v=')[1]
    logging.info("Process: Video Download")
    downloadYouTube(youtube_link, f_name=video_id)
    logging.info("Download: Video Completed")

    #top_k = 5 이미지 top 이므로 현재 단계에서 사용X

    filepath = f'videos/{video_id}.mp4'
    video, fps = get_video_info(filepath)

    
    # save 구조로 갈지 returen 구조로 갈지 고민 필요
    save_video_frame(filepath, video, fps, 1) # 마지막은 몇 초에 한번 프레임을 생성할 것인지
    image_list = natsorted(glob.glob(f'{filepath[:-4]}/*.jpg'))
    #
    threshold = 75
    min_scene_len= fps*10 # fps * 시간
    scene_list = detect(filepath, ContentDetector(min_scene_len=min_scene_len, threshold=threshold))
    
    scene_time = [[int(x[0].get_frames()/fps), int(x[1].get_frames()/fps)] for x in scene_list]
    print(scene_time)
    
    # 추가
    np.save(f'videos/{video_id}.npy', scene_time)
    #
    txt_file_path = f'videos/{video_id}.txt'

    llm_reult_path = f'videos/{video_id}_llm_score.npy'
    if wait_for_file(llm_reult_path):
        text = load_txt_summarize(video_id)
        llm_result = np.load(llm_reult_path)

        model_name = 'openai/clip-vit-base-patch32'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CLIPModel.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}').to(device)
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}')
        batch_size = 64

    
    score_tensor, class_name = get_visual_scores(model, processor, device, text, image_list, batch_size)
    #get_top_frame(score_tensor, image_list, filepath, top_k)
    final_interval_all, final_interval_vision, final_interval_llm = split_frame(scene_time, class_name, score_tensor, llm_result)
    
    print("ALL RESEULT\n",final_interval_all)
    print('-' * 100)
    print("VISION RESEULT\n",final_interval_vision)
    print('-' * 100)
    print("LLM RESEULT\n",final_interval_llm)
    print('-' * 100)

    dict_file = {"video_id": video_id, "table": 3, "interval_":final_interval_all}
    send_data_to_server(dict_file,server)
    
    with conn.cursor() as cursor:
        for idx, inter_ in enumerate(final_interval_all):
            f = f"""INSERT INTO querys.TIME_INTER (youtube_id, topic_idx, interval_) VALUES ("{video_id}", "{idx+1}", "{json.dumps(inter_)}")"""
            cursor.execute(f)
        conn.commit()

    with conn.cursor() as cursor:
        for idx, inter_ in enumerate(final_interval_all):
            f = f"""INSERT INTO youtube.TIME_INTER (youtube_id, topic_idx, interval_) VALUES ("{video_id}", "{idx+1}", "{json.dumps(inter_)}")"""
            cursor.execute(f)
        conn.commit()

    with conn.cursor() as cursor:
        f = f'DELETE FROM querys.WEB WHERE youtube_id = "{video_id}"'
        cursor.execute(f)
        conn.commit()
