import logging
import librosa

from youtube2audio import downloadYouTubeAudio
from audio2text import convertAudio2Text
from utils import audioUtils
import json
import threading


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

def process(youtube_link):
    logging.info(f"Process: {youtube_link}")
    video_id=youtube_link.split('watch?v=')[1][:11]

    logging.info("Process: Audio Download")
    downloadYouTubeAudio(video_id,youtube_link)
    logging.info("Download: Audio Completed")

    logging.info("Process: Loading Audio")
    sample, sampling_rate = librosa.load(f'../videos/{video_id}.mp3')
    logging.info("Process: Audio to Text")
    script, timestamps = convertAudio2Text(sample,video_id)

    logging.info("Download: Audio Script Completed")
    duration=audioUtils.get_audio_duration(video_id)
    audioUtils.get_norm_timestamp_json(duration, video_id)
    script, timestamps = audioUtils.get_audio_text_json(video_id)
    logging.info("Download: Time-Normalized Audio Script Completed")
    return video_id


def write_requests2(video_id):
    with open('../requests2.json', 'r+', encoding='UTF-8') as file2:
        try:
            requests = json.load(file2)
        except:
            requests = []
        requests.append({"video_id": video_id})
        json.dump(requests, file2)

def check_request_loop():
    while True:
        with open('../requests1.json', 'r+', encoding='UTF-8') as file:
            try:
                requests = json.load(file)
                if requests:
                    request = requests.pop(0)
                    link=request['link']
                    time_script = process(link)
                    file.seek(0)
                    json.dump(requests, file)
                    file.truncate()
                    write_requests2(link, time_script)
            except:
                pass

def check_requests():
    with open('../requests1.json', 'r+', encoding='UTF-8') as file:
        requests = json.load(file)
        if requests:
            request = requests.pop(0)
            link=request['link']
            video_id = process(link)
            file.seek(0)
            json.dump(requests, file)
            file.truncate()
            write_requests2(video_id)


if __name__ == "__main__":
    request_thread = threading.Thread(target=check_requests)
    request_thread.start()
    request_thread.join()
