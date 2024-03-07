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
def get_visual_scores(text, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def get_top_frame(score_tensor, filepath, top_k):    
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


def split_frame(filepath, class_name, score_tensor): # AdaptiveDetector, ContentDetector 중 선택
    # AdaptiveDetector
    #scene_list = detect(filepath, AdaptiveDetector())

    # ContentDetector
    threshold = 80
    min_scene_len= 29.97*10 # fps * 시간
    scene_list = detect(filepath, ContentDetector(min_scene_len=min_scene_len, threshold=threshold))
    
    print(f'Splitting video by {len(scene_list)}... it takes time')
    split_video_ffmpeg(filepath, scene_list)
    scene_time = [[int(x[0].get_frames()/fps), int(x[1].get_frames()/fps)] for x in scene_list]
    print(scene_time)
    scene_score = [score_tensor[x[0]:x[1]] for x in scene_time]
    scene_score_class = torch.stack([x.mean(dim=0) for x in scene_score])

    k = 3 #len(scene_time)
    for sentence in range(score_tensor.shape[1]):
        top_k_value, top_k_index = torch.topk(scene_score_class[:, sentence], k, largest=True)

        print(f"Class: {class_name[sentence]}:")
        print([(int(x.to('cpu')+1),float(y.to('cpu'))) for x,y in zip(top_k_index,top_k_value)])

    return scene_score_class


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    batch_size = 64
    top_k = 5

    filepath = 'data/video/test3.mp4'
    image_list = natsorted(glob.glob(f'{filepath[:-4]}/*.jpg'))
    video, fps = get_video_info(filepath)

    # 다른 모듈과 일치 시키기 위해 바꿔줘야함
    text = """
    1. Mack's encounter with the thousand spiders: This scene is a classic fear-based challenge that is sure to entertain audiences as Mack faces his arachnophobia in a high-pressure situation. The suspense builds as he tries to remain calm and complete the challenge, making it a crowd-pleaser.
    2. Mack's trust fall: In this scene, Mack's friendship with his partner is put to the test as he must trust him to catch him during a blind jump off a plank. The potential for disaster adds to the tension and excitement, making it an entertaining moment.
    3. Mack's struggle to save the duffel bags: This action-packed scene is full of suspense as Mack races against the clock to save as many bags of money as possible from a sinking car. The danger and difficulty of the task make it an exciting and entertaining moment.
    4. Mack's reaction to the Feastables bars: This lighthearted scene provides a nice contrast to the more intense challenges as Mack tries a new product and shares his thoughts on the new formula and flavors. The unexpected twist and Mack's honest reactions make it an entertaining moment.
    5. Mack's experience being buried alive: This nerve-wracking and entertaining moment tests Mack's mental strength as he faces his fear of being buried alive and struggles to keep track of time and deal with his claustrophobia. The psychological aspect of the challenge adds an extra layer of intrigue and suspense.
    """

    # save 구조로 갈지 returen 구조로 갈지 고민 필요
    save_video_frame(filepath, video, fps, 1) # 마지막은 몇 초에 한번 프레임을 생성할 것인지
    score_tensor, class_name = get_visual_scores(text, batch_size)
    #get_top_frame(score_tensor, filepath, top_k)
    scene_score_class = split_frame(filepath, class_name, score_tensor)
