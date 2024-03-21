import logging
import pickle
import librosa
import json
import re
import os
from settings import *
from youtube2audio import downloadYouTubeVideo, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text
from utils import scores, videoUtils, audioUtils, domainFlow, llmUtils
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # 0 for first GPU

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

seed = 2024
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"


#원래는 domainFlow.py에 위치
def separate_reduce_common_domain(domain, text):
    prompt=prompt_reduce_common_pick2()
    return summarize_mapreduce(text, prompt)
#원래는prompt.py위치
def prompt_reduce_common_pick1():
    """
    Reduce prompt for all common domains to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top 5 scenes that could be the most important moments or 'hot clips' from various parts of the script.
    The answer should follows format below.
    
    Important Moments: {ext_sum}

    <>[/INST]<\s>.
    """
    return template

def prompt_reduce_common_pick2():
    """
    Reduce prompt for all common domains to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top 5 scenes that could be the most highlight moments or 'hot clips' from various parts of the script.
    The answer should follows format below.
    
    Highlight Moments: {ext_sum}

    <>[/INST]<\s>.
    """
    return template

def prompt_reduce_common_pick3():
    """
    Reduce prompt for all common domains to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top 5 scenes that could be the most viewed moments or 'hot clips' from various parts of the script.
    The answer should follows format below.

    Viewed Moments: {ext_sum}

    <>[/INST]<\s>.
    """
    return template



if __name__ == '__main__':
    
    with open('heatmap_20.json' , 'r') as f:
        heatmap = json.load(f)
        
    for topic in heatmap:
        for i,info in enumerate(heatmap[topic]):
            youtube_link,title,duration,svg,timestamps,script_time = info
            video_id=youtube_link.split('watch?v=')[1]
            print(topic, title)

            with timer('Summariazation part'):
                logging.info("Process: Text Summarization")
                torch.cuda.empty_cache()
                output_llm=separate_reduce_common_domain(topic, script_time)
                # output_llm=separate_reduce_common_domain(topic, script_time)
                print(output_llm)
                heatmap[topic][i].append(output_llm)
                with open('heatmap_llm.json', 'w', encoding='utf-8') as file:
                    json.dump(heatmap, file, ensure_ascii=False, indent=4)
            if i >20:
                break
        #         # input “요약문 리스트 [문장1, 문장2, 문장3, 문장4, 문장5]”
        #         matches = re.findall(r'\d+\..*?\n', output_llm)
        #         summarization = [match.split('.', 1)[1].strip() for match in matches]

        #         llmUtils.save_txt_summarize(output_llm,video_id)
        #         logging.info("Download: Summarized txt Completed")

        #     with timer('Retrieve part'):  
        #         logging.info("Process: Summ-Text top-k candidates")
        #         # CV section load
        #         logging.info("CV section load")
        #         #section = np.load(f'videos/{video_id}.npy')

        #         output_llm=llmUtils.load_txt_summarize(video_id)
        #         txt_llm=llmUtils.get_origin_text(output_llm)

        #         timeselect=[]
        #         #timecand=scores.make_candidates(timestamps,section)
        #         timecand=scores.make_candidates(timestamps)
        #         script = []
        #         for idx in timecand:
        #             script.append(idx[0])

        #         logging.info("Process: Retrieve top-k timeline")
        #         Vectorizer = TfidfVectorizer().fit(script)
        #         corpusVec = Vectorizer.transform(script)

        #         results = []
        #         for txt in txt_llm:
        #             txt_vec =Vectorizer.transform([txt])
        #             value = np.dot(corpusVec,txt_vec.T).toarray().flatten()

        #         llm_score = np.empty((0,len(script)))
        #         for r_origin_topic in txt_llm:
        #             r_origin_vec =Vectorizer.transform([r_origin_topic])
        #             value = np.dot(corpusVec,r_origin_vec.T).toarray().flatten()
        #             llm_score =np.append(llm_score,np.array([value]),axis=0)
                
        #         results = []
        #         for num in range(llm_score.shape[0]):
        #             piv = np.argmax(llm_score[num])
        #             results.append(timecand[piv][1:])
        #         print(results)
        #         #np.save(f'videos/{video_id}_llm_score.npy', llm_score)
        #     break
        # break