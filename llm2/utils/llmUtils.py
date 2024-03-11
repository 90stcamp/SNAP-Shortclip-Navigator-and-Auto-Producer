import json
import re


def save_txt_summarize(text,video_id):
    with open(f'../videos/{video_id}.txt', 'w', encoding="UTF-8") as file:
        text+="/n/n"
        file.write(text)

def load_txt_summarize(video_id):
    with open(f'../videos/{video_id}.txt', 'r', encoding="UTF-8") as file:
        text = file.read()
    return text

def get_origin_text(text):
    matches = re.findall(r'\d+\..*?\n', text)
    responds = [match.split('.', 1)[1].strip() for match in matches]
    return responds

def save_text_json(output,video_id):
    with open(f'../videos/{video_id}.json', 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=4)

def change_timestamp_list(video_id):
    """
    Change timestamp json for llm list format
    """
    with open(f'../videos/{video_id}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    array=[]
    for item in data["chunks"]:
        array.append(f"({item['timestamp'][0]}) {item['text']}")
    return " \n ".join(array)

def get_audio_text_json(video_id):
    with open(f'../videos/{video_id}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    return data['text'], data['chunks']