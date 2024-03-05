import logging
import pickle
import librosa
import json

from youtube2audio import downloadYouTube, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text
from utils import scores, videoutils, audioutils, crawlers, domainflow


if __name__ == '__main__':
    logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)
    logging.info("Process Started")
    youtube_link='https://www.youtube.com/watch?v=KrLj6nc516A'
    video_id=youtube_link.split('watch?v=')[1]
    category=crawlers.get_youtube_category(youtube_link)
    logging.info(f"Video Category: {category}")

    logging.info("Video Download Process")
    downloadYouTube(youtube_link, f_name=video_id)
    logging.info("Video to Audio Process")
    convertVideo2Audio(f"videos/{video_id}.mp4")

    logging.info("Loading Audio Process")
    sample, sampling_rate = librosa.load(f'videos/{video_id}.mp3')    
    logging.info("Audio to Text Process")
    script, timestamps = convertAudio2Text(sample,video_id)
    duration=audioutils.get_audio_duration(video_id)
    audioutils.get_norm_timestamp_json(duration, video_id)
    script_time = audioutils.change_timestamp_list(video_id)

    logging.info("Text Summarization Process")
    response_list=[]
    response=domainflow.separate_reduce_domain(category, script_time)
    if len(response) > 0:
        response_list.append([response, script_time])
    text=[timestamps,response]

    logging.info("Summ-Text top-k retrieval Process")
    candidates = scores.top_k_text(text,60,3,1) 
    videoutils.cut_video(video_id,candidates)

