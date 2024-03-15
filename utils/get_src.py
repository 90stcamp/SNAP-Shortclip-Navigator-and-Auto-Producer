import subprocess
import os
import re

def downloadYouTubeVideo(video_id, videourl):
    command = ['yt-dlp', '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', '--no-write-auto-subs', '--sub-lang', 'en', '--write-sub', '-o', f'{video_id}/{video_id}.%(ext)s', videourl]
    subprocess.run(command)
    remove_video_name(video_id)

    exist_vtt = check_subtitle(video_id)
    #함수를 통해 .vtt(자막) 이 존재하는지 판별합니다.
    return exist_vtt
    #존재한다면 Timestamps를 아니면 False를 반환합니다.

def remove_video_name(video_id):
    for filename in os.listdir(video_id):
        if filename.startswith(video_id):
            old_path = os.path.join(video_id, filename)
            if filename.endswith('.mp4'):
                new_filename=f"{video_id}.mp4"
            elif filename.endswith('.m4a'):
                new_filename=f"{video_id}.m4a"
            elif filename.endswith('.vtt'):
                new_filename=f"{video_id}.vtt"
            new_path = os.path.join(video_id, new_filename)
            os.rename(old_path, new_path)    

def check_subtitle(video_id):
    vtt = os.path.join(video_id, f"{video_id}.vtt")
    if os.path.exists(vtt):
        timestamps = parse_vtt_to_json(vtt)
        return timestamps
    else:
        return False

def parse_vtt_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 메타데이터 건너뛰기
        next(file)
        next(file)
        next(file)
        
        vtt_content = file.readlines()

    pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})')
    result = []
    text = ''
    start_seconds = 0
    end_seconds = 0
    
    for line in vtt_content:
        if pattern.match(line):
            if text:  # 이전 대사가 있으면 결과에 추가
                result.append({
                    "timestamp": [start_seconds, end_seconds],
                    "text": text.strip().replace('\n', ' ')
                })
                text = ''  # 텍스트 초기화
            start, end = pattern.match(line).groups()
            start_seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], start.split(":")))
            end_seconds = sum(x * float(t) for x, t in zip([3600, 60, 1], end.split(":")))
        elif line.strip():
            # '-'로 시작하는 대사의 시작 기호를 제거
            clean_line = line.lstrip('- ')
            text += clean_line 

    if text:
        result.append({
            "timestamp": [start_seconds, end_seconds],
            "text": text.strip().replace('\n', ' ')
        })

    return result

timestamps =downloadYouTubeVideo('KOEfDvr4DcQ','https://www.youtube.com/watch?v=KOEfDvr4DcQ')

if timestamps:
    #직접달은 영어자막이 존재함
    pass
else:
    #존재안함
    pass