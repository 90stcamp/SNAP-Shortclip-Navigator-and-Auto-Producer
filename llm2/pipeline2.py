import logging

from text2summ import *
from utils import scores, domainFlow, llmUtils, crawlers
import json
import threading


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

def process(video_id):
    logging.info("Process2 Started")
    category=crawlers.get_youtube_category(video_id)
    logging.info(f"Video Category: {category}")

    script, timestamps = llmUtils.get_audio_text_json(video_id)
    script_time = llmUtils.change_timestamp_list(video_id)

    logging.info("Process: Text Summarization")
    output_llm=domainFlow.separate_reduce_domain(category, script_time)
    txt_llm=llmUtils.get_origin_text(output_llm)
    llmUtils.save_txt_summarize(output_llm,video_id)
    logging.info("Download: Summarized txt Completed")

    logging.info("Process: Summ-Text top-k candidates")
    output_llm=llmUtils.load_txt_summarize(video_id)
    timeselect=[]
    timecand=scores.make_candidates(timestamps,20,1)

    logging.info("Process: Retrieve top-k timeline")
    for res in txt_llm:
        timeselect.append(scores.retrieve_top_k_cosine(res, timecand))
    llmUtils.save_text_json(timeselect,video_id+"_top")
    logging.info("Download: Summarized txt Completed")
    return timeselect


def write_requests3(video_id, timeselect):
    with open('../requests3.json', 'r+', encoding='UTF-8') as file2:
        try:
            requests = json.load(file2)
        except:
            requests = []
        requests.append({video_id: timeselect})
        json.dump(requests, file2)

def check_request_loop():
    while True:
        with open('../requests2.json', 'r+', encoding='UTF-8') as file:
            try:
                requests = json.load(file)
                if requests:
                    request = requests.pop(0)
                    video_id = request['video_id']
                    timestamp = process(video_id)
                    file.seek(0)
                    json.dump(requests, file)
                    file.truncate()
                    write_requests3(video_id, timestamp)
            except:
                pass
            
def check_requests():
    with open('../requests2.json', 'r+', encoding='UTF-8') as file:
        requests = json.load(file)
        if requests:
            request = requests.pop(0)
            video_id = request['video_id']
            timestamp = process(video_id)
            file.seek(0)
            json.dump(requests, file)
            file.truncate()
            write_requests3(video_id, timestamp)


if __name__ == "__main__":
    request_thread = threading.Thread(target=check_requests)
    request_thread.start()
    request_thread.join()
