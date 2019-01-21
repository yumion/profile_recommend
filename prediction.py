import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_preprocessing.image import img_to_array, list_pictures, load_img
#from ncc.preprocessing import preprocess_input

dataset = 'owndata/'

def img_load_from_dir(target_dir):
    x_array = []
    pictures = glob(target_dir+'*/*')
    classnames = [name.split('/')[-1] for name in glob(target_dir+'*')]
    class_indices = []
    for picture in pictures:
        # 入力 x
        img = load_img(picture) # img type = PIL.image
        img_array = img_to_array(img) # np.array
        x_array.append(img_array.astype('uint8')) # input image
        # 教師ラベルy
        classname = picture.split('/')[-2]
        class_indices.append(int(classnames.index(classname)))
    # ndarrayに変換
    x_array = np.asarray(x_array)
    return x_array, class_indices

def resize_img_array(img_array, height=224, width=224):
    temp_array = []
    imgs_size = []
    for i in range(len(img_array)):
        temp_array.append(cv2.resize(img_array[i], (width, height)))
    img_array = np.asarray(temp_array)
    return img_array

# 読み込み
x_origin, y_test = img_load_from_dir(dataset)
# preprocessing
x_test = resize_img_array(x_origin)
#x_test = preprocess_input(x_test)
x_test = x_test.astype('float32')
x_test /= 255. 
print(x_origin[0].shape)
# クラス名をとってくる
class_names = [x.split('/')[-1] for x in glob(dataset+'*')]
print(class_names)

# modelのロード
model_cls = load_model('google_seasons_VGGaug_weights_add_train.h5')
# model_value = load_model('rank_model.h5')

## どの画像がどのクラスへ分類されたかを保存
class_pred_onehot = model_cls.predict(x_test) #クラス推定
# rank_pred_onehot = model_value.predict(x_test) #ランク推定
# one-hotから数字へ
class_pred = np.argmax(class_pred_onehot, axis=1)
# rank_pred = np.argmax(rank_pred_onehot, axis=1)
# print(class_pred.shape, rank_pred.shape)
print(class_pred)

# フォルダに保存
for idx, (pred_cls_idx, true_cls_idx) in enumerate(zip(class_pred, y_test)):
    cv2.imwrite('prediction/{pred}/{true}_{cnt:04d}.jpg'.format(pred=class_names[pred_cls_idx], true=class_names[true_cls_idx], cnt=idx), x_origin[idx][..., ::-1])
