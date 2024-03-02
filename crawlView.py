from selenium import webdriver
from bs4 import BeautifulSoup
import time


options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)



driver.get("https://www.youtube.com/watch?v=VHre-G8wlb4")


soup = BeautifulSoup(driver.page_source, 'html.parser')

progress_bar = soup.find("div", {"id": "progress-bar-id"})

if progress_bar:
    viewed_percentage = progress_bar.text

driver.quit()

print(viewed_percentage)
