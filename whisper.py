from transformers import AutoProcessor, pipeline, AutoModelForSpeechSeq2Seq
import librosa 
import argparse 

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

if __name__ == '__main__':
    sample, sampling_rate = librosa.load('videos/bbc_news.mp3')
    
    processor = AutoProcessor.from_pretrained(MODEL_DICT['whisper-large'], cache_dir='/data/ephemeral')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_DICT['whisper-large'], low_cpu_mem_usage=True, cache_dir='/data/ephemeral').to('cuda:0')
    
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        return_timestamps='word',
        device='cuda:0'
    )
    
    output = pipe(sample)
    
