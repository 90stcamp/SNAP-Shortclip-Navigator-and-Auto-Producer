for domain in news docu show movie
do
    ffmpeg -i videos/$domain.mp4 -vn -y -q:a 0 videos/$domain.mp3
done
