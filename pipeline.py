import logging
import pickle
import librosa
import json

from youtube2audio import downloadYouTube, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text
from utils import scores, videoedit, crawlers, domainflow


if __name__ == '__main__':
    logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)
    logging.info("Process Started")
    youtube_link='https://www.youtube.com/watch?v=6_PI1l5NKL8'
    video_dir=youtube_link.split('watch?v=')[1]
    category=crawlers.get_youtube_category(youtube_link)
    logging.info(f"Video Category: {category}")

    logging.info("Video Download Process")
    downloadYouTube(youtube_link, f_name=video_dir)
    logging.info("Video to Audio Process")
    convertVideo2Audio(f"videos/{video_dir}.mp4")

    logging.info("Loading Audio Process")
    sample, sampling_rate = librosa.load(f'videos/{video_dir}.mp3')    
    logging.info("Audio to Text Process")
    script, timestamps = convertAudio2Text(sample,video_dir)

    with open(f'videos/{video_dir}.json', 'r', encoding='utf-8') as json_file:
        output = json.load(json_file)
    script, timestamps = output['text'][3001:6000], output['chunks']

    logging.info("Text Summarization Process")
    response_list=[]
    # To set len_sen=0 and script 
    response=domainflow.separate_to_domain(category,script)
    if len(response) > 0:
        response_list.append([response, script])
    text =[timestamps,response]

    # print(response)
    # logging.info("Summ-Text top-k retrieval Process")
    # candidates = scores.top_k_text(text,60,3,1) #shorts길이 , Top K , candidates 간격
    # videoedit.cut_video(video_dir,candidates)

