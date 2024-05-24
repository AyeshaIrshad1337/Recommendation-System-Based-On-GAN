
import pandas as pd
from selenium  import webdriver
import time
import os
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def fetch_movie_posters(movie_titles, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    options = Options()
    options.headless = True
    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    for title in movie_titles:
        try:
            search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={title} movie poster"
            driver.get(search_url)
            time.sleep(2)  # Allow time for images to load
            
            images = driver.find_elements(By.CSS_SELECTOR, 'img')
            if images:
                image_url = images[0].get_attribute('src')
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                if img.mode in ("RGBA", "LA", "P"):
                        img = img.convert("RGB")
                img.save(os.path.join(output_dir, f'{title}.jpg'))
                print(f'Successfully downloaded poster for "{title}"')
            else:
                print(f'No image found for "{title}"')
        except Exception as e:
            print(f'Failed to download poster for "{title}": {e}')
    
    driver.quit()
if __name__ == '__main__':
    movie=pd.read_csv('movies.csv')
    movie_titles = movie['title']
    fetch_movie_posters(movie_titles,'images')