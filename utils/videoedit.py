from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_merge_video_audio
import subprocess
import os


def cut_video(video_dir,candidates):
    for i,final_candidate in enumerate(candidates):
        start_time, end_time = final_candidate[1], final_candidate[2]
        ffmpeg_extract_subclip(f'videos/{video_dir}.mp4', start_time, end_time, targetname='temp.mp4')
        ffmpeg_extract_subclip(f'videos/{video_dir}.mp3', start_time, end_time, targetname='temp.mp3')
        #start,end timed으로 자르기
        cmd = f"ffmpeg -i temp.mp4 -i temp.mp3 -c:v copy -c:a mp3 -map 0:v:0 -map 1:a:0 videos/{video_dir[:3]}_shorts_{i+1}.mp4"
        #ffmpeg 커맨드 , temp.mp4에 temp.mp3 합침 , 같은거 방지 오류 {video_dir[:3]}
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove('temp.mp4')
        os.remove('temp.mp3')
        #임시파일삭제
