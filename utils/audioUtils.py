import librosa
import json


def get_audio_duration(video_id):
    filename = f'videos/{video_id}.mp3'
    y, sr = librosa.load(filename, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

def get_audio_text_json(video_id):
    with open(f'videos/{video_id}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    return data['text'], data['chunks']

def get_norm_timestamp_json(duration, video_id):
    """
    To normalize whisper output timestamp
    """
    with open(f'videos/{video_id}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)

    # for the case that last value of timestamp endpoint returns null
    if not data['chunks'][-1]['timestamp'][-1]:
        data['chunks'][-1]['timestamp'][-1]=data['chunks'][-1]['timestamp'][0]+2
    rate=duration/data['chunks'][-1]['timestamp'][-1]

    for i in range(len(data['chunks'])):
        data['chunks'][i]['timestamp'][0]=round(data['chunks'][i]['timestamp'][0]*rate,2)
        data['chunks'][i]['timestamp'][1]=round(data['chunks'][i]['timestamp'][1]*rate,2)

    with open(f'videos/{video_id}.json', 'w', encoding="UTF-8") as file:
        json.dump(data, file, indent=4)

def save_text_json(output,video_id):
    with open(f'videos/{video_id}.json', 'w', encoding='utf-8') as json_file:
        json.dump(output, json_file, indent=4)

def change_timestamp_list(video_id):
    """
    Change timestamp json for llm list format
    """
    with open(f'videos/{video_id}.json', 'r', encoding="UTF-8") as file:
        data = json.load(file)
    array=[]
    for item in data["chunks"]:
        array.append(f"({item['timestamp'][0]}) {item['text']}")
    return " \n ".join(array)
