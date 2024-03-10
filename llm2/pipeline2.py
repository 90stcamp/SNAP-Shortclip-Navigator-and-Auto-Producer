import logging
import librosa

from text2summ import *
from utils import scores, domainFlow, llmUtils, crawlers
import json
import threading


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

def process(video_id, ):


if __name__ == '__main__':
    logging.info("Process2 Started")

    video_id=youtube_link.split('watch?v=')[1]
    category=crawlers.get_youtube_category(youtube_link)
    logging.info(f"Video Category: {category}")

    script_time = audioUtils.change_timestamp_list(video_id)

    logging.info("Process: Text Summarization")
    output_llm=domainFlow.separate_reduce_domain(category, script_time)
    llmUtils.save_txt_summarize(output_llm,video_id)
    logging.info("Download: Summarized txt Completed")

    logging.info("Process: Summ-Text top-k candidates")
    output_llm=llmUtils.load_txt_summarize(video_id)
    txt_llm=llmUtils.get_origin_text(output_llm)
    timeselect=[]
    timecand=scores.make_candidates(timestamps,20,1)

    logging.info("Process: Retrieve top-k timeline")
    for res in txt_llm:
        timeselect.append(scores.retrieve_top_k_cosine(res, timecand))
    llmUtils.save_text_json(timeselect,video_id+"_top")
    logging.info("Download: Summarized txt Completed")

def check_requests():
    while True:
        with open('../requests1.json', 'r+', encoding='UTF-8') as file:
            try:
                requests = json.load(file)
                if requests:
                    link = requests.pop(0)['link']
                    time_script = process(link)
                    file.seek(0)
                    json.dump(requests, file)
                    file.truncate()
                    write_requests2(link, time_script)
            except:
                pass

if __name__ == "__main__":
    request_thread = threading.Thread(target=check_requests)
    request_thread.start()
    request_thread.join()
