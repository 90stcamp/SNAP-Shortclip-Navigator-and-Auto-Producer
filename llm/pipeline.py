import logging
import pickle
import librosa
import json
import re

from settings import *
from youtube2audio import downloadYouTubeVideo, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text
from utils import scores, videoUtils, audioUtils, domainFlow, llmUtils
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dbs
import requests
from requests.exceptions import Timeout, RequestException

def send_data_to_server(dict_file,server):
    response = requests.post(server, json=dict_file, timeout=10)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

    
if __name__ == '__main__':
    # youtube_id = "KOEfDvr4DcQ"
    # category="Entertainment"
    conn = dbs.MYSQL_DATABASE_CONN

    with open(SERVER_DIR, "r") as env:
        dic_server = json.load(env)
    server = dic_server['server']

    youtube_link = os.getenv('YOUTUBE_LINK')
    logging.info("Process Started")
    video_id=youtube_link.split('watch?v=')[1]

    category = os.getenv("YOUTUBE_CATEGORY")


    with conn.cursor() as cursor:
        f = f"""INSERT INTO querys.WEB (youtube_id, category) VALUES ("{video_id}", "{category}")"""
        cursor.execute(f)
        conn.commit()

    with conn.cursor() as cursor:
        f = f"""INSERT INTO youtube.WEB (youtube_id, category) VALUES ("{video_id}", "{category}")"""
        cursor.execute(f)
        conn.commit()


    with timer('Whisper Part'):
        logging.info(f"Video Category: {category}")

        logging.info("Process: Video Download")
        exist_subtitle = downloadYouTubeVideo(video_id, videourl=youtube_link)
        logging.info("Download: Video Completed")
        logging.info("Process: Video to Audio")

        if exist_subtitle:
            timestamps = exist_subtitle
            script_time = audioUtils.change_timestamp_list_for_exist(timestamps)
        
        else:

            convertVideo2Audio(f"videos/{video_id}.mp4")
            logging.info("Download: Audio Completed")

            logging.info("Process: Loading Audio")
            sample, sampling_rate = librosa.load(f'videos/{video_id}.mp3')    
            logging.info("Process: Audio to Text")
            script, timestamps = convertAudio2Text(sample,video_id)
            logging.info("Download: Audio Script Completed")

            duration=audioUtils.get_audio_duration(video_id)
            audioUtils.get_norm_timestamp_json(duration, video_id)
            script, timestamps = audioUtils.get_audio_text_json(video_id)

    
            logging.info("Download: Time-Normalized Audio Script Completed")
            script_time = audioUtils.change_timestamp_list(video_id)
        # 완료 신호
        dict_file = {"video_id": video_id, "table": 1}
        send_data_to_server(dict_file,server)

    with timer('Summariazation part'):
        logging.info("Process: Text Summarization")
        output_llm=domainFlow.separate_reduce_domain(category, script_time)
        print(output_llm)
        # input “요약문 리스트 [문장1, 문장2, 문장3, 문장4, 문장5]”
        matches = re.findall(r'\d+\..*?\n', output_llm)
        summarization = [match.split('.', 1)[1].strip() for match in matches]

        # 완료 신호
        dict_file = {"video_id": video_id, "table": 2}
        send_data_to_server(dict_file,server)
        
        for idx, summ in enumerate(summarization):
            # summ = summ.replace('"', "\'")
            summ = re.sub('"', "\'", summ)
            with conn.cursor() as cursor:
                f = f"""INSERT INTO querys.LLM (youtube_id, topic_idx, summarization) VALUES ("{video_id}", "{idx+1}", "{summ}")"""
                print(f)
                cursor.execute(f)
                conn.commit()

        for idx, summ in enumerate(summarization):
            summ = re.sub('"', "\'", summ)
            with conn.cursor() as cursor:
                f = f"""INSERT INTO youtube.LLM (youtube_id, topic_idx, summarization) VALUES ("{video_id}", "{idx+1}", "{summ}")"""
                cursor.execute(f)
                conn.commit()

        llmUtils.save_txt_summarize(output_llm,video_id)
        logging.info("Download: Summarized txt Completed")

    with timer('Retrieve part'):  
        logging.info("Process: Summ-Text top-k candidates")
        # CV section load
        logging.info("CV section load")
        section = np.load(f'videos/{video_id}.npy')

        output_llm=llmUtils.load_txt_summarize(video_id)
        txt_llm=llmUtils.get_origin_text(output_llm)

        timeselect=[]
        timecand=scores.make_candidates(timestamps,section)
        script = []
        for idx in timecand:
            script.append(idx[0])

        logging.info("Process: Retrieve top-k timeline")
        Vectorizer = TfidfVectorizer().fit(script)
        corpusVec = Vectorizer.transform(script)

        results = []
        for txt in txt_llm:
            txt_vec =Vectorizer.transform([txt])
            value = np.dot(corpusVec,txt_vec.T).toarray().flatten()

        llm_score = np.empty((0,len(script)))
        for r_origin_topic in txt_llm:
            r_origin_vec =Vectorizer.transform([r_origin_topic])
            value = np.dot(corpusVec,r_origin_vec.T).toarray().flatten()
            llm_score =np.append(llm_score,np.array([value]),axis=0)
        print(llm_score.shape)

        np.save(f'videos/{video_id}_llm_score.npy', llm_score)