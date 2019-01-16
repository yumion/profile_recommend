import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import time

pages = 2000
values = ['春', '夏', '秋', '冬']
folders = ['spring','summer', 'autumn', 'winter']

for value, folder in zip(values, folders):
    cnt = 0

    for page in range(1, pages+1):

        url = 'https://photohito.com/tag/{value}/?o=popular-all&p={page}'.format(value='春', page=1)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        image_urls = soup.find_all('a', href=re.compile('/photo/'))

        for image_url in image_urls[3::2][:-1]: # 画像とテキストに同じリンクが貼ってあってだぶるため１つ飛ばし.最後のリンクは除く
            time.sleep(0.5) # 高速にアクセスするとサーバーダウンさせる危険あるため時間おく
            # 画像の個別ページへ飛ぶ
            img_page = requests.get('http://photohito.com'+image_url.get('href'))
            img_soup = BeautifulSoup(img_page.text, 'lxml')

            # お気に入り登録数をとってくる
            temp_soup = img_soup.find_all('a', href=re.compile('/user/register'))[2] # なぜかtext=re.compile("お気に入り登録")では引っかからないので、ルールベースで
            num_favo = temp_soup.span.string # お気に入り登録数
            num_view = img_soup.find_all('li', id='act_view')[0].string # 閲覧数

            # タイトルをとってくる
            img_url = img_soup.find_all('div', id='photo_view')[0]
            title = img_url.h1.string
            # 画像をとってくる
            img_src = img_url.find_all('img', alt=title)[0]
            r = requests.get(img_src.get('src'))
            img = Image.open(BytesIO(r.content))

            # 時々画像のリンクが切れている場合があるためそれを回避
            if img.mode == "RGB":
                # img.show()
                img.save('dataset/{folder}/{cnt}_{title}_{season}_like{like}_views{view}.jpg'.format(folder=folder, cnt=cnt, title=title, season=value, like=num_favo, view=num_view), 'JPEG', quality=100, optimize=True)
            # elif img.mode != "RGB":
                # img = img.convert("RGB")
            cnt += 1
        # 指定した枚数でストップ
        if cnt > 10000:
            break
print('finished')
