import librosa

path="auSvQtj4BqE"
# Load the audio file
filename = f'/home/jaealways/Youtube-Short-Generator/videos/{path}.mp3'
y, sr = librosa.load(filename, sr=None)  # Load the file as is without resampling for accurate duration
duration_in_seconds = librosa.get_duration(y=y, sr=sr)

print(f"The duration of the audio file is: {duration_in_seconds} seconds")


import json

file_name="{path}.json"
with open(f'videos/{file_name}', 'r', encoding="UTF-8") as file:
    data = json.load(file)
rate=duration_in_seconds/data['chunks'][-1]['timestamp'][-1]

for i in range(len(data['chunks'])):
    data['chunks'][i]['timestamp'][0]=data['chunks'][i]['timestamp'][0]*rate
    data['chunks'][i]['timestamp'][1]=data['chunks'][i]['timestamp'][1]*rate


# Writing the JSON data to a file
with open(f'videos/{path}_norm.json', 'w', encoding="UTF-8") as file:
    json.dump(data, file, indent=4)  # indent for pretty printing

