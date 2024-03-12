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
    class_ = text.split('\n') # 다른 모듈과 일치 시키기 위해 바꿔줘야함
    class_name = class_[1:-1]
    images = [Image.open(image) for image in image_list]

    # 빈 텐서 생성
    score_tensor = torch.empty(0).to(device)

    print('Measuring Similarity...')
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i + batch_size]
        inputs = processor(text=class_name, images=batch, return_tensors="pt", padding=True)

        outputs = model(**inputs.to(device))
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        # 스코어를 텐서에 추가
        score_tensor = torch.cat((score_tensor, probs.detach()), dim=0)
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


def split_frame(filepath, class_name, score_tensor, fps, split): # AdaptiveDetector, ContentDetector 중 선택
    # AdaptiveDetector
    #scene_list = detect(filepath, AdaptiveDetector())

    # ContentDetector
    threshold = 80
    min_scene_len= fps*10 # fps * 시간
    scene_list = detect(filepath, ContentDetector(min_scene_len=min_scene_len, threshold=threshold))
    
    scene_time = [[int(x[0].get_frames()/fps), int(x[1].get_frames()/fps)] for x in scene_list]
    print(scene_time)
    scene_score = [score_tensor[x[0]:x[1]] for x in scene_time]
    scene_score_class = torch.stack([x.mean(dim=0) for x in scene_score])

    print(f'Splitting video by {len(scene_list)}... it takes time')
    if not os.path.exists('videos/result/'):
        os.makedirs('videos/result/')
    k = 3 #len(scene_time)

    final_interval = []
    if split:
        for sentence in tqdm(range(score_tensor.shape[1])):
            top_k_value, top_k_index = torch.topk(scene_score_class[:, sentence], k, largest=True)
            print(f"Class: {class_name[sentence]}:")
            print([(itemgetter(x)(scene_time),float(y)) for x,y in zip(top_k_index,top_k_value)])
            final_interval.append([itemgetter(x)(scene_time) for x in top_k_index])
            split_video_ffmpeg(filepath, itemgetter(*top_k_index)(scene_list), video_name=f"videos/result/{filepath.split('/')[-1][:-4]}-{sentence+1}")
    else: 
        for sentence in tqdm(range(score_tensor.shape[1])):
            top_k_value, top_k_index = torch.topk(scene_score_class[:, sentence], k, largest=True)
            print(f"Class: {class_name[sentence]}:")
            print([(itemgetter(x)(scene_time),float(y)) for x,y in zip(top_k_index,top_k_value)])
            final_interval.append([itemgetter(x)(scene_time) for x in top_k_index])
            #split_video_ffmpeg(filepath, itemgetter(*top_k_index)(scene_list), video_name=f"videos/result/{filepath.split('/')[-1][:-4]}-{sentence+1}")

    return final_interval

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

if __name__=='__main__':
    youtube_link = input("Input your youtube video link: ")
    #text = input("Input the scenario you'd like to create for short: ")

    logging.info("Process Started")
    video_id=youtube_link.split('watch?v=')[1]
    logging.info("Process: Video Download")
    downloadYouTube(youtube_link, f_name=video_id)
    logging.info("Download: Video Completed")
    model_name = 'openai/clip-vit-base-patch32'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}').to(device)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=f'models/{model_name.split("/")[-1]}')
    batch_size = 64
    #top_k = 5 이미지 top 이므로 현재 단계에서 사용X

    filepath = f'videos/{video_id}.mp4'
    video, fps = get_video_info(filepath)

    # save 구조로 갈지 returen 구조로 갈지 고민 필요
    save_video_frame(filepath, video, fps, 1) # 마지막은 몇 초에 한번 프레임을 생성할 것인지
    image_list = natsorted(glob.glob(f'{filepath[:-4]}/*.jpg'))

    # 다른 모듈과 일치 시키기 위해 바꿔줘야함(json/parser로?)
    # text = get_origin_text(text)
    text = """
    1. Mack's encounter with the thousand spiders: This scene is a classic fear-based challenge that is sure to entertain audiences as Mack faces his arachnophobia in a high-pressure situation. The suspense builds as he tries to remain calm and complete the challenge, making it a crowd-pleaser.
    2. Mack's trust fall: In this scene, Mack's friendship with his partner is put to the test as he must trust him to catch him during a blind jump off a plank. The potential for disaster adds to the tension and excitement, making it an entertaining moment.
    3. Mack's struggle to save the duffel bags: This action-packed scene is full of suspense as Mack races against the clock to save as many bags of money as possible from a sinking car. The danger and difficulty of the task make it an exciting and entertaining moment.
    4. Mack's reaction to the Feastables bars: This lighthearted scene provides a nice contrast to the more intense challenges as Mack tries a new product and shares his thoughts on the new formula and flavors. The unexpected twist and Mack's honest reactions make it an entertaining moment.
    5. Mack's experience being buried alive: This nerve-wracking and entertaining moment tests Mack's mental strength as he faces his fear of being buried alive and struggles to keep track of time and deal with his claustrophobia. The psychological aspect of the challenge adds an extra layer of intrigue and suspense.
    """
    
    score_tensor, class_name = get_visual_scores(model, processor, device, text, image_list, batch_size)
    #get_top_frame(score_tensor, image_list, filepath, top_k)
    final_interval = split_frame(filepath, class_name, score_tensor, fps, split=False)
    print(final_interval)
    # return 후 파일 삭제(output video를 넘겨준 뒤 없애야하므로 추후 수정)
    shutil.rmtree('videos/')
