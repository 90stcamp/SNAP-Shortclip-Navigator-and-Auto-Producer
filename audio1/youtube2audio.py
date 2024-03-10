import subprocess


def downloadYouTubeAudio(video_id, videourl):
    command = ['yt-dlp', '-x', '--audio-format', 'mp3', '-o', f"../videos/{video_id}", videourl]
    subprocess.run(command)