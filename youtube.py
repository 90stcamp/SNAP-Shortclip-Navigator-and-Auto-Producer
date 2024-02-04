import argparse 
from pytube import YouTube
import os

def downloadYouTube(videourl, f_name, path='videos'):
    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(output_path=path, filename=f'{f_name}.mp4')


DOMAIN = {
    'news': 'https://www.youtube.com/watch?v=e0V1CtzFgrU', 
    'docu': 'https://www.youtube.com/watch?v=rKtzj_9vl8Q', 
    'show': 'https://www.youtube.com/watch?v=6_PI1l5NKL8', 
    'movie': 'https://www.youtube.com/watch?v=BxHvI5BVBf4'
}


parser = argparse.ArgumentParser()

parser.add_argument('--domain', '-d', required=True, help='must be news, docu, show, and movie')

args = parser.parse_args()

if __name__ == '__main__':
    video_path = DOMAIN[args.domain]
    downloadYouTube(video_path, f_name=f'{args.domain}')
