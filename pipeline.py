import librosa 
import logging

from youtube2audio import downloadYouTube, convertVideo2Audio
from text2summ import *
from audio2text import convertAudio2Text


def input_llm(len_sen,doc):
    template=prompts.prompt_extsum_paper2()
    prompt = PromptTemplate(template=template, input_variables=["len_sen","document"])
    llm_chain = LLMChain(prompt=prompt, llm=hf)
    response = llm_chain.invoke(input={"len_sen": len_sen, "document": doc})
    torch.cuda.empty_cache()
    return response['text']


if __name__ == '__main__':
    logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.INFO)
    logging.info("Process Started")
    youtube_link='https://www.youtube.com/watch?v=6Pm0Mn0-jYU&ab_channel=CNBCMakeIt'
    video_dir=youtube_link.split('watch?v=')[1]

    logging.info("Video Download Process")
    downloadYouTube(youtube_link, f_name=video_dir)
    logging.info("Video to Audio Process")
    convertVideo2Audio(f"videos/{video_dir}.mp4")

    logging.info("Loading Audio Process")
    sample, sampling_rate = librosa.load(f'videos/{video_dir}.mp3')    
    logging.info("Audio to Text Process")
    script, timestamps = convertAudio2Text(sample)

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    cache_dir = "/data/ephemeral/Youtube-Short-Generator/models/mistral"
    
    logging.info("Text Summarization Process")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir)
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=100, 
        device = 0, 
        pad_token_id=tokenizer.eos_token_id)
    hf = HuggingFacePipeline(pipeline=pipe)

    response_list = []
    len_sen=0
    response = input_llm(len_sen, script)
    if len(response) > 0:
        response_list.append([response, script])
    print(response_list)

