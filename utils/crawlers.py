# from selenium import webdriver
from bs4 import BeautifulSoup
import time

# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.common.by import By


def get_heatmap_html(url):
    driver = webdriver.Chrome()
    driver.get(url)

    bar=driver.find_element(By.CSS_SELECTOR, "div.ytp-progress-bar-container")
    action = ActionChains(driver)
    # perform the operation
    action.move_to_element(bar).click().perform()
    bar.click()

    # Get the page source once the element is available
    graph=driver.find_element(By.XPATH, "svg/defs/clipPath")

    # Extract the 'd' attribute from each selected element
    d_values = [element.get('d') for element in graph]
    return d_values


def get_heatmap_data():
    css_heat_map_path = "ytp-heat-map-path"
    heat_map_data = get_heatmap_html("https://www.youtube.com/watch?v=279KL-eVnVM")
    return heat_map_data


def get_youtube_category(youtube_link):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'user-agent={user_agent}')    
    driver = webdriver.Chrome(options=options)

    driver.get(f'{youtube_link}')
    html=driver.page_source
    category=html.split('"category":"')[1].split('",')[0]
    return category


TopicList=["Animation and Film", "Autos and Vehicles", "Music Videos", "Pets and Animals",
           "Sports", "Travel & Events", "Gaming", "People and Blogs", "Comedy",
           "Entertainment", "News and Politics", "How to And Style", "Education",
           "Science And Technology", "NonProfit & Activism"]


if __name__=='__main__':
    youtube_link="https://www.youtube.com/watch?v=7mkvRjAhNOo"
    category=get_youtube_category(youtube_link)
    print(category)
