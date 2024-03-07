from selenium import webdriver
from bs4 import BeautifulSoup
import time
import json
from pprint import pprint
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from tqdm import tqdm

TopicList=["Animation and Film", "Autos and Vehicles", "Music Videos", "Pets and Animals",
           "Sports", "Travel & Events", "Gaming", "People and Blogs", "Comedy",
           "Entertainment", "News and Politics", "How to And Style", "Education",
           "Science And Technology", "NonProfit & Activism"]
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument(f'user-agent={user_agent}')    
driver = webdriver.Chrome(options=options)

keyword = "Comedy"  # 검색 키워드

def convert_to_seconds(time_str):
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 3:
        # 시간, 분, 초가 모두 있는 경우
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        # 분, 초만 있는 경우
        return parts[0] * 60 + parts[1]
    else:
        return parts[0]
    
def get_youtube_category(youtube_link,length):
    
    #카테고리
    driver.get(f'{youtube_link}')
    time.sleep(3)
    try:
        html=driver.page_source
        category=html.split('"category":"')[1].split('",')[0]

        soup = BeautifulSoup(html, 'html.parser')
        #라이브 동영상은 건너뛰기
        script = soup.find('script', text=lambda t: t and 'ytInitialPlayerResponse' in t)
        json_str = script.string.split('ytInitialPlayerResponse = ')[1].split(';var meta')[0]
        data = json.loads(json_str)
        live_status = data['videoDetails']['isLiveContent']
        #카테고리 다르다면 건너뛰기
        if live_status or keyword not in category:
            return 0,0
        #제목
        title = soup.select_one('title').text
        print(youtube_link)
        print(title, category,length)
    except:
        return 0,0
    return category, title

def crawl_youtube(keyword, num_urls=10):
    driver.get('https://www.youtube.com/results?search_query=' + keyword + '&sp=CAMSAhAB')  # 조회수 순 정렬 URL

    url_list = []
    time_list = []
    while len(url_list) < num_urls:
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        video_elements = soup.select('#video-title')
        time_elements = soup.select('.ytd-thumbnail-overlay-time-status-renderer#text')
        for video, times in zip(video_elements, time_elements):
            url = video.get('href')
            print(url)
            # url이 인식이 안되면 건너뛰기(광고) ,이미 있는 url이면 건너뛰기. shorts라면 건너뛰기 
            if url==None or url in url_list or 'shorts' in url or 'SHORTS'in times.text.strip():
                continue
            #시간이 10분이상 30분이하가 아니라면 건너뛰기
            time_secs = convert_to_seconds(times.text.strip())
            if 600<=time_secs<=1800:
                url_list.append(url)
                time_list.append(time_secs)  # 영상 길이 정보를 가져와서 리스트에 추가
        # 페이지를 스크롤합니다.
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3.0)

    # URL 앞에 도메인을 붙여 완전한 URL을 만듬
    full_urls = ['https://www.youtube.com' + url for url in url_list]

    return full_urls[:num_urls], time_list[:num_urls]



if __name__=='__main__':
    urls,lengths = crawl_youtube(keyword, 10)
    real_urls= []
    for i in tqdm(range(len(urls))):
        category,title=get_youtube_category(urls[i],lengths[i])
        if category in keyword :
            real_urls.append(urls[i])
    pprint(real_urls)