import logging
import pickle
import librosa
import json

from youtube2audio import downloadYouTube, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text
from utils import scores, videoutils, audioutils, crawlers, domainflow


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

def processYoutube(link):
    video_id=link.split('watch?v=')[1]
    category=crawlers.get_youtube_category(link)

    logging.info(f"Video Category: {category}")

    logging.info("Video Download Process")
    downloadYouTube(youtube_link, f_name=video_id)
    logging.info("Video to Audio Process")
    convertVideo2Audio(f"videos/{video_id}.mp4")
    return video_id, category

def processAudio2Text(video_id):
    logging.info("Loading Audio Process")
    sample, sampling_rate = librosa.load(f'videos/{video_id}.mp3')    
    logging.info("Audio to Text Process")
    script, timestamps = convertAudio2Text(sample,video_id)
    duration=audioutils.get_audio_duration(video_id)
    audioutils.get_norm_timestamp_json(duration, video_id)
    script_time = audioutils.change_timestamp_list(video_id)
    return script, timestamps, script_time

def processTextSumm(category, script_time):
    logging.info("Text Summarization Process")
    response_list=[]
    response=domainflow.separate_reduce_domain(category, script_time)
    if len(response) > 0:
        response_list.append([response, script_time])
    return [timestamps,response]

def processTextTime(text):
    logging.info("Summ-Text top-k retrieval Process")
    candidates = scores.top_k_text(text,60,3,1) 
    videoutils.cut_video(video_id,candidates)
    return 


if __name__ == '__main__':
    logging.info("Process Started")
    youtube_link='https://www.youtube.com/watch?v=KOEfDvr4DcQ'
    video_id, category=processYoutube(youtube_link)
    # script, timestamps, script_time=processAudio2Text(video_id)
    # text=processTextSumm(category, script_time)
    # processTextTime(text)

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
    script, timestamps = audioutils.get_audio_text_json(video_id)
    duration=audioutils.get_audio_duration(video_id)
    audioutils.get_norm_timestamp_json(duration, video_id)
    script_time = audioutils.change_timestamp_list(video_id)

    logging.info("Text Summarization Process")
    response=domainflow.separate_reduce_domain(category, script_time)
    utils.save_txt_summarize(response,video_id)

    response=utils.load_txt_summarize(video_id)
    script, timestamps = audioutils.get_audio_text_json(video_id)
    text=[timestamps,response]

    logging.info("Summ-Text top-k retrieval Process")
    candidates = scores.top_k_text(text,60,3,1)
    print(candidates)
    videoutils.cut_video(video_id,candidates)

