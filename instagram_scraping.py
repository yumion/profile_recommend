import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import time

pages = 1000
value = '冬'
folder = 'winter'

cnt = 0
for page in range(1, pages+1):

    url = 'https://photohito.com/search/photo/?value={value}&camera-maker=0&camera-model=0&lens-maker=0&lens-model=0&focallength_from=0&focallength_to=0&pref=0&area=0&year=0&month=0&day=0&range=0&order=popular-all&p={page}'.format(value=value, page=page)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    image_urls = soup.find_all('a', href=re.compile('http://photohito.com/photo/')) # 画像の個別ページへいく

    for image_url in image_urls[1::2]: # 画像とテキストに同じリンクが貼ってあってだぶるため１つ飛ばし
        time.sleep(0.5)
        img_page = requests.get(image_url.get('href'))
        img_soup = BeautifulSoup(img_page.text, 'lxml')

        temp_soup = img_soup.find_all('a', href=re.compile('/user/register'))[2] # なぜかtext=re.compile("お気に入り登録")では引っかからないので、ルールベースで
        num_favo = temp_soup.span.string # お気に入り登録数

        img_url = img_soup.find_all('img')[3] # 他のおすすめ画像も下に出るので回避
        r = requests.get(img_url.get('src'))
        img = Image.open(BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        # img.show()
        img.save('dataset/{folder}/{cnt:04d}_like{like}.jpg'.format(folder=folder, cnt=cnt, like=num_favo), 'JPEG', quality=100, optimize=True)
        cnt += 1
    if cnt > 3000: break

'''
import json
import random


def login_instagram(username, passwd):
    BASE_URL = 'https://www.instagram.com/accounts/login/'
    LOGIN_URL = BASE_URL + 'ajax/'

    headers_list = [
            "Mozilla/5.0 (Windows NT 5.1; rv:41.0) Gecko/20100101"\
            " Firefox/41.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2)"\
            " AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2"\
            " Safari/601.3.9",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0)"\
            " Gecko/20100101 Firefox/15.0.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"\
            " (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36"\
            " Edge/12.246"
            ]

    USERNAME = username
    PASSWD = passwd
    USER_AGENT = headers_list[random.randrange(0,4)]

    session = requests.Session()
    session.headers = {'user-agent': USER_AGENT}
    session.headers.update({'Referer': BASE_URL})
    req = session.get(BASE_URL)
    soup = BeautifulSoup(req.content, 'html.parser')
    body = soup.find('body')

    pattern = re.compile('window._sharedData')
    script = body.find("script", text=pattern)

    script = script.get_text().replace('window._sharedData = ', '')[:-1]
    data = json.loads(script)

    csrf = data['config'].get('csrf_token')
    login_data = {'username': USERNAME, 'password': PASSWD}
    session.headers.update({'X-CSRFToken': csrf})
    login = session.post(LOGIN_URL, data=login_data, allow_redirects=True)
    login.content
    return session

s = login_instagram(username='yumion7488', passwd='yushimo28')
'''

'''
from instabot import Bot

bot = Bot()
bot.login(username='yumion7488', password='yushimo28')

hashtag = '春'
num_likes
medias = bot.get_hashtag_medias(hashtag, filtration=False)
bot.download_photos(medias, folder=hashtag)
'''
