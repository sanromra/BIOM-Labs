"""
Acknoledgement to @petercour. Script originally extracted from https://dev.to/petercour/image-scraping-with-python-330a
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os

"""
Download images from baidu based on a keyword
"""
def download_baidu(keyword): 
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+word+'&ct=201326592&v=flip'
    result = requests.get(url)
    html = result.text
    pic_url = re.findall('"objURL":"(.*?)",',html,re.S)
    i = 0

    for each in pic_url:
        print(pic_url)
        try:
            pic= requests.get(each, timeout=10)
        except requests.exceptions.ConnectionError:
            print ('exception')
            continue

        string = 'pictures'+keyword+'_'+str(i) + '.jpg'
        fp = open(string,'wb')
        fp.write(pic.content)
        fp.close()
        i += 1


"""
Download images from Google Images based on a keyword
"""
def download_google(word):
    url = 'https://www.google.com/search?q=' + word + '&client=opera&hs=cTQ&source=lnms&tbm=isch&sa=X&ved=0ahUKEwig3LOx4PzKAhWGFywKHZyZAAgQ_AUIBygB&biw=1920&bih=982'
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')

    for raw_img in soup.find_all('img'):
       link = raw_img.get('src')
       os.system("wget " + link)


if __name__ == '__main__':
    word = input("Input key word: ") #Request keyword on standard input
    download_baidu(word) #Search Baidu
    download_google(word) #Search Google Images
