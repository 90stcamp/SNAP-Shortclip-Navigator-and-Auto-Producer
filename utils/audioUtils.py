import librosa
import json


def get_audio_duration(path):
    filename = f'videos/{path}.mp3'
    y, sr = librosa.load(filename, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

def get_norm_timestamp_json(duration, path):
    """
    To normalize whisper output timestamp
    """
    with open(f'videos/{path}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    rate=duration/data['chunks'][-1]['timestamp'][-1]

    for i in range(len(data['chunks'])):
        data['chunks'][i]['timestamp'][0]=round(data['chunks'][i]['timestamp'][0]*rate,2)
        data['chunks'][i]['timestamp'][1]=round(data['chunks'][i]['timestamp'][1]*rate,2)

    with open(f'videos/{path}.json', 'w', encoding="UTF-8") as file:
        json.dump(data, file, indent=4)

def save_text_json(output,path):
    with open(f'videos/{path}.json', 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=4)

def change_timestamp_list(path):
    """
    Change timestamp json for llm list format
    """
    with open(f'videos/{path}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    array=[]
    for item in data["chunks"]:
        array.append(f"({item['timestamp'][0]}) {item['text']}")
    return " \n ".join(array)
