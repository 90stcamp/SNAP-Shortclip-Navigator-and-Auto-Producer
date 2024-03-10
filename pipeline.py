import logging
import librosa

from youtube2audio import downloadYouTube, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text
from utils import scores, videoUtils, audioUtils, crawlers, domainFlow, llmUtils


logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s', 
                    datefmt ='%m/%d %I:%M:%S %p', level=logging.INFO)


if __name__ == '__main__':
    # youtube_link=input()
    youtube_link="https://www.youtube.com/watch?v=KOEfDvr4DcQ"
    logging.info("Process Started")

    video_id=youtube_link.split('watch?v=')[1]
    category=crawlers.get_youtube_category(youtube_link)
    # category="Entertainment"
    logging.info(f"Video Category: {category}")

    logging.info("Process: Video Download")
    downloadYouTube(youtube_link, f_name=video_id)
    logging.info("Download: Video Completed")
    logging.info("Process: Video to Audio")
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
    audioUtils.save_text_json(timeselect,video_id+"_top")
    logging.info("Download: Summarized txt Completed")
    videoUtils.cut_video(video_id,timeselect)

