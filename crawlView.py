from selenium import webdriver
from bs4 import BeautifulSoup
import time

from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By


css_heat_map_path = "ytp-heat-map-path"

def get_html(url):
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


# Example usage:
heat_map_data = get_html("https://www.youtube.com/watch?v=279KL-eVnVM")
print(heat_map_data)
