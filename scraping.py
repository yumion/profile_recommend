import requests
from bs4 import BeautifulSoup
import re
from PIL import Image
from io import BytesIO
import time

pages = 2000
value = '秋'
folder = 'autumn'

cnt = 3007
for page in range(189, pages+1):

    url = 'https://photohito.com/search/photo/?value={value}&camera-maker=0&camera-model=0&lens-maker=0&lens-model=0&focallength_from=0&focallength_to=0&pref=0&area=0&year=0&month=0&day=0&range=0&order=popular-all&p={page}'.format(value=value, page=page)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    image_urls = soup.find_all('a', href=re.compile('http://photohito.com/photo/')) # 画像の個別ページへいく

    for image_url in image_urls[1::2]: # 画像とテキストに同じリンクが貼ってあってだぶるため１つ飛ばし
        time.sleep(0.5) # 高速にアクセスするとサーバーダウンさせる危険あるため時間おく
        # 画像の個別ページへ飛ぶ
        img_page = requests.get(image_url.get('href'))
        img_soup = BeautifulSoup(img_page.text, 'lxml')
        # お気に入り登録数をとってくる
        temp_soup = img_soup.find_all('a', href=re.compile('/user/register'))[2] # なぜかtext=re.compile("お気に入り登録")では引っかからないので、ルールベースで
        num_favo = temp_soup.span.string # お気に入り登録数
        # 画像をとってくる
        img_url = img_soup.find_all('img')[3] # 他のおすすめ画像も下に出るので回避
        r = requests.get(img_url.get('src'))
        img = Image.open(BytesIO(r.content))
        # 時々画像のリンクが切れている場合があるためそれを回避
        if img.mode == "RGB":
            # img.show()
            img.save('dataset/{folder}/{cnt:04d}_like{like}.jpg'.format(folder=folder, cnt=cnt, like=num_favo), 'JPEG', quality=100, optimize=True)
        # elif img.mode != "RGB":
            # img = img.convert("RGB")
        cnt += 1
    # 指定した枚数でストップ
    if cnt > 10000:
        break

from tkinter import Tk, messagebox
root = Tk()
root.withdraw()
messagebox.showinfo(value, str(cnt)+'枚')  # 情報ダイアログ表示
root.quit()
