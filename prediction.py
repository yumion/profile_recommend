import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_preprocessing.image import img_to_array, list_pictures, load_img


def img_load_from_dir(target_dir):
    x_array = []
    pictures = glob(target_dir)
    for picture in list_pictures(pictures):
        # 入力 x
        img = load_img(picture) # img type = PIL.image
        img_array = img_to_array(img) # np.array
        x_array.append(img_array.astype('uint8')) # input image
    # ndarrayに変換
    x_array = np.asarray(x_array)
    return x_array

x_test = img_load_from_dir('test_data/')

class_names = [x.split('/')[-1] for x in glob('dataset/*')] # クラス名をとってくる
# print(class_names)

# modelのロード
model_cls = load_model('seasons_model.h5')
model_value = load_model('rank_model.h5')

## どの画像がどのクラスへ分類されたかを保存
class_pred_onehot = model_cls.predict(x_test)
# print(class_pred_onehot)
# one-hotから数字へ
class_pred = np.argmax(class_pred_onehot, axis=1)
# print(class_pred.shape)

## ランクを推定
rank_pred_onehot = model_value.predict(x_test)
# print(rank_pred_onehot)
# one-hotから数字へ
rank_pred = np.argmax(rank_pred_onehot, axis=1)
# print(rank_pred.shape)

# フォルダに保存
for idx, class_idx, rank in zip(enumerate(class_pred), rank_pred):
    cv2.imwrite('prediction/{class}/{rank}_{cnt:04d}.jpg'.format(class=class_names[class_idx], rank=rank, cnt=idx), x_test[idx])
