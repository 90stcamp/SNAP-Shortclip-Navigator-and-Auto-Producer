from transformers import AutoProcessor, pipeline, AutoModelForSpeechSeq2Seq
import librosa 
import argparse 
import spacy
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

import gc
import torch


MODEL_DICT = {
    'wav2vec': 'facebook/wav2vec2-base-960h', 
    'wav2vec-bert': 'hf-audio/wav2vec2-bert-CV16-en', 
    'univnet': 'dg845/univnet-dev', 
    's2t': 'facebook/s2t-small-librispeech-asr', 
    'whisper-tiny': 'openai/whisper-tiny.en', 
    'whisper-medium': 'openai/whisper-medium', 
    'whisper-large': 'openai/whisper-large-v2'
}

parser = argparse.ArgumentParser(description='speech')
parser.add_argument(
    '--model_name', required=True, type=str
)

def get_sentence_timestamps(result):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(result['text'])
    sentences = [sent.text for sent in doc.sents]
    timestamps, words = [], result['chunks']
    count=0

    for sen in sentences:
        listWords = sen.split()
        for i in range(len(listWords)):
            timestamp = words[count]['timestamp'][0]
            if i == 0:
                start_time = timestamp
            end_time = words[count]['timestamp'][1]
            count+=1
        timestamps.append((sen, start_time, end_time))
    return timestamps


def convertAudio2Text(sample):
    processor = AutoProcessor.from_pretrained(MODEL_DICT['whisper-large'], cache_dir='models')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_DICT['whisper-large'], cache_dir='models').to('cuda:0')
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        device='cuda:0'
    )
    
    output = pipe(sample, return_timestamps=True)
    timestamps=get_sentence_timestamps(output)

    return output['text'], timestamps


def convertAudio2TextJax(audio_path):
    video_path=audio_path.replace('mp4','mp3')
    sample, sampling_rate = librosa.load(video_path)
    pipeline=FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16)
    result=pipeline(sample, return_timestamps=True)
    return result


def flush():
    gc.collect()
    torch.cuda.empty_cache()


# for default
if __name__ == '__main__':
    sample, sampling_rate = librosa.load('videos/6Pm0Mn0-jYU&ab_channel=CNBCMakeIt.mp3')    
    output = convertAudio2Text(sample)
    print(output)


# for whisper jax
# if __name__ == '__main__':
#     flush()
#     output = convertAudio2TextJax('videos/6Pm0Mn0-jYU&ab_channel=CNBCMakeIt.mp3')
#     print(output)

