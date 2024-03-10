import logging
import librosa

from youtube2audio import downloadYouTube, convertVideo2Audio
from audio2text import convertAudio2Text
from utils import audioUtils, crawlers
import time
import json
import threading


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

def process(youtube_link):
    logging.info(f"Process: {youtube_link}")

    video_id=youtube_link.split('watch?v=')[1]
    # category=crawlers.get_youtube_category(youtube_link)
    # logging.info(f"Video Category: {category}")

    # logging.info("Process: Video Download")
    # downloadYouTube(youtube_link, f_name=video_id)
    # logging.info("Download: Video Completed")
    # logging.info("Process: Video to Audio")
    # convertVideo2Audio(f"videos/{video_id}.mp4")
    # logging.info("Download: Audio Completed")

    # logging.info("Process: Loading Audio")
    # sample, sampling_rate = librosa.load(f'videos/{video_id}.mp3')
    # logging.info("Process: Audio to Text")
    # script, timestamps = convertAudio2Text(sample,video_id)

    # logging.info("Download: Audio Script Completed")
    # duration=audioUtils.get_audio_duration(video_id)
    # audioUtils.get_norm_timestamp_json(duration, video_id)
    # script, timestamps = audioUtils.get_audio_text_json(video_id)

    # logging.info("Download: Time-Normalized Audio Script Completed")
    # script_time = audioUtils.change_timestamp_list(video_id)
    # print(script_time)


def check_requests():
    while True:
        with open('../requests.json', 'r+', encoding='UTF-8') as file:
            requests = json.load(file)
            if requests:
                current_request = requests.pop(0)
                process(current_request['link'])
                file.seek(0)
                json.dump(requests, file)
                file.truncate()
            time.sleep(5)

if __name__ == "__main__":
    request_thread = threading.Thread(target=check_requests)
    request_thread.start()
    request_thread.join()




