from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import re
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_preprocessing.image import img_to_array, list_pictures, load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.objectives import categorical_crossentropy, mse
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from ncc.validations import evaluate
from ncc.history import show_history

class MyGenerator(Sequence):
    """Custom generator"""
    def __init__(self, data_paths, data_classes, num_of_class=4, batch_size=32, width=299, height=299, ch=3):
        """construction
        :param data_paths: List of image file
        :param data_classes: List of class
        :param batch_size: Batch size
        :param width: Image width
        :param height: Image height
        :param ch: Num of image channels
        :param num_of_class: Num of classes
        """
        self.data_paths = data_paths
        self.data_classes = data_classes
        self.length = len(data_paths)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.ch = ch
        self.num_of_class = num_of_class
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1


    def __getitem__(self, idx):
        """Get batch data
        :param idx: Index of batch
        :return imgs: numpy array of images
        :return labels: numpy array of label
        """
        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        item_paths = self.data_paths[start_pos : end_pos]
        item_classes = self.data_classes[start_pos : end_pos]

        imgs = np.zeros((len(item_paths), self.height, self.width, self.ch), dtype=np.float32)
        labels = np.zeros((len(item_paths), self.num_of_class))
        for i, (item_path, item_class) in enumerate(zip(item_paths, item_classes)):
            img, label = self._load_data(item_path, item_class, self.num_of_class, self.width, self.height)
            imgs[i, :] = img
            labels[i, label] = 1.
        return imgs, labels

    def __len__(self):
        """Batch length"""
        return self.num_batches_per_epoch

    def on_epoch_end(self):
        """Task when end of epoch"""
        pass

    def _load_data(self, item_path, item_class, num_of_class, width, height):
        # 入力 x
        img = load_img(item_path) # img type = PIL.image
        img_array = img_to_array(img) # np.array
        img_array = cv2.resize(img_array, (height, width))
        img_array /= 255.
        # ラベル y
        label = int(item_class)
        return img_array, label


# get class name
class_names = [x.split('/')[-1] for x in glob('dataset/*')] # クラス名をとってくる
print(class_names)

def load_img_path_and_class(target_dir):
    paths = glob(target_dir+'*/*')
    classnames = [name.split('/')[-1] for name in glob(target_dir+'*')]
    class_indices = []
    for path in paths:
        classname = path.split('/')[-2]
        class_indices.append(int(classnames.index(classname)))
    return paths, class_indices

def load_num_likes_from_dir(target_dir):
    y_array = []
    pictures = glob(target_dir)
    regex = re.compile(r'like(.*).jpg')
    for picture in pictures:
        # ラベル y
        mo = regex.search(picture) # ファイル名からお気に入り数を取得
        num_likes = mo.group(1)
        y_array.append(int(num_likes)) # label
    return y_array

# お気に入り数をランク分けに変換
def convert_to_class(y_likes):
    for idx, likes in enumerate(y_likes):
        if likes > 300:
            y_likes[idx] = 2
        elif likes > 200 and likes <= 300:
            y_likes[idx] = 1
        elif likes <= 200:
            y_likes[idx] = 0
    return y_likes


x_paths, y_classes = load_img_path_and_class('dataset/')
y_likes = load_num_likes_from_dir('dataset/*/*')

y_likes = convert_to_class(y_likes)

# お気に入り数の分布
# print(np.max(y_array), np.argmax(y_array))
# print(len(y_array))
num_likes = np.zeros(np.max(y_likes)+1)
for like in y_likes:
    num_likes[like] += 1
plt.bar(np.arange(len(num_likes)), num_likes)
plt.xlabel('Number of favorite')
plt.ylabel('Number of count')
# plt.xlim(0, 1100)
# plt.ylim(0,200)
# plt.savefig('distribution_favorite.png')
plt.show()

# split train and test
# train_paths, test_paths, train_classes, test_classes, train_likes, test_likes = train_test_split(x_paths, y_classes, y_likes, test_size=0.1, random_state=1225)

# input data profile
season_classes = len(class_names) # 4 season
rank_classes = np.max(y_likes)+1
# input_shape = len(train_paths)
# print(input_shape)
print(rank_classes)
print(season_classes)

'''
# 画像枚数が多いのでgeneratorで渡す
# 季節分類
season_train_gen = MyGenerator(train_paths, train_classes, num_of_class=season_classes, batch_size=128)
season_test_gen = MyGenerator(test_paths, test_classes, num_of_class=season_classes, batch_size=128)
# お気に入り数分類
rank_train_gen = MyGenerator(train_paths, train_likes, num_of_class=rank_classes, batch_size=128)
rank_test_gen = MyGenerator(test_paths, test_likes, num_of_class=rank_classes, batch_size=128)
'''

"ImageDataGenerator"
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

season_train_gen = datagen.flow_from_directory(
        'dataset',
        target_size=(299, 299),
        batch_size=128,
        class_mode='categorical',
        subset='training')

season_test_gen = datagen.flow_from_directory(
        'dataset',
        target_size=(299, 299),
        batch_size=128,
        class_mode='categorical',
        subset='validation')


def rankGenerator(target_dir, subset=None, train_ratio=0.1, batch_size=32, height=299, width=299):
    file_list = glob(target_dir+'*/*')
    regex = re.compile(r'like(.*).jpg')
    # train test split
    train_size = len(file_list) * (1-train_ratio)
    train_size = int(train_size)
    if subset == 'training':
        split_file_list = file_list[:train_size]
        np.random.shuffle(split_file_list)
    elif subset == 'validation':
        split_file_list = file_list[train_size:]
        np.random.shuffle(split_file_list)
    else:
        split_file_list = file_list

    for img_path in split_file_list:
        x_array, y_array = [], []
        # image
        img = load_img(img_path) # img type = PIL.image
        img_array = img_to_array(img) # np.array
        img_array = cv2.resize(img_array, (height, width))
        img_array /= 255.
        x_array.append(img_array)
        # ラベル y
        mo = regex.search(img_path) # ファイル名からお気に入り数を取得
        num_likes = mo.group(1)
        y_array.append(int(num_likes))
        if len(x_array) == batch_size:
            y_array = convert_to_class(y_array)
            x_array = np.asarray(x_array)
            y_array = np.asarray(y_array)
            yield x_array, y_array


rank_train_gen = rankGenerator('dataset/', batch_size=128, subset='training')
rank_test_gen = rankGenerator('dataset/', batch_size=128, subset='validation')


"""Inception v3"""
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# classification
x_cls = Dense(1024, activation='relu')(x)
x_cls = Dropout(0.25)(x_cls)
x_cls = Dense(256, activation='relu')(x_cls)
x_cls = Dropout(0.5)(x_cls)
classification = Dense(season_classes, activation='softmax', name='classification')(x_cls)
# regression
x_rgs = Dense(1024, activation='relu')(x)
x_rgs = Dropout(0.25)(x_rgs)
x_rgs = Dense(256, activation='relu')(x_rgs)
x_rgs = Dropout(0.5)(x_rgs)
regression = Dense(rank_classes, activation='softmax', name='regression')(x_rgs)

## 季節分類
# this is the model we will train
# del model_cls
model_cls = Model(inputs=base_model.input, outputs=classification)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model_cls.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# train the model on the new data for a few epochs
history_cls = model_cls.fit_generator(season_train_gen,
                steps_per_epoch=len(season_train_gen),
                validation_data=season_test_gen,
                validation_steps=len(season_test_gen),
                epochs=3,
                shuffle=True
                )

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
   # print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model_cls.layers[:249]:
    layer.trainable = False
for layer in model_cls.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model_cls.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# callbacks = [EarlyStopping(patience=5)]
history_cls_add = model_cls.fit_generator(season_train_gen,
                steps_per_epoch=len(season_train_gen),
                validation_data=season_test_gen,
                validation_steps=len(season_test_gen),
                epochs=30,
                shuffle=True
                )

# save and eval model
# evaluate(model_cls, x_test, cls_test, class_names)
# save_show_results(history, model)
json_model_cls = model_cls.to_json()
open('seasons_model.json', 'w').write(json_model_cls)
model_cls.save_weights('seasons_weight_add.h5')
# show_history(history_cls)
acc = history_cls.history['acc'] + history_cls_add.history['acc']
val_acc = history_cls.history['val_acc'] + history_cls_add.history['val_acc']
plt.plot(range(33), acc, label='acc')
plt.plot(range(33), val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
# plt.savefig('season_classifier.png')
plt.show()

## お気に入り数の回帰
# クラスに重み付け
print(num_likes)
ratio_likes = num_likes / np.max(num_likes) # 正規化
ratio_likes = np.reciprocal(ratio_likes) # 逆数
class_weight = {}
for idx, value in enumerate(ratio_likes):
    class_weight[idx] = value
print(class_weight)

# del model_value
model_value = Model(inputs=base_model.input, outputs=regression)

for layer in base_model.layers:
    layer.trainable = False

model_value.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# train the model on the new data for a few epochs
history_value = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=len(rank_train_gen),
                validation_data=rank_test_gen,
                validation_steps=len(rank_test_gen),
                epochs=1,
                class_weight=class_weight,
                shuffle=True
                )

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model_value.layers[:249]:
    layer.trainable = False
for layer in model_value.layers[249:]:
    layer.trainable = True

from keras.optimizers import SGD
model_value.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
# callbacks = [EarlyStopping(patience=5)]
history_value_add = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=len(rank_train_gen),
                validation_data=rank_test_gen,
                validation_steps=len(rank_test_gen),
                epochs=10,
                class_weight=class_weight,
                shuffle=True
                )
# save and eval model
# evaluate(model_value, x_test, value_test)
# show_history(history_value)
acc = history_value.history['acc'] + history_value_add.history['acc']
val_acc = history_value.history['val_acc'] + history_value_add.history['val_acc']
plt.plot(range(33), acc, label='acc')
plt.plot(range(33), val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('favo_classifier.png')
plt.show()
model_value.save('rank_model_add.h5')


# prediction
from keras.models import model_from_json
del model_cls
model_cls = load_model('seasons_model_add.h5')
# json_string = open('seasons_model.json').read()
# model_cls = model_from_json(json_string)
# model_cls.load_weights('seasons_weight_add.h5')

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
season_test_gen = datagen.flow_from_directory(
        'dataset',
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='validation')
# どの画像がどのクラスへ分類されたかを保存
y_pred = model_cls.predict_generator(season_test_gen, steps=len(season_test_gen))
print(y_pred)
class_pred = np.argmax(y_pred, axis=1)
print(class_pred)
import collections
collections.Counter(class_pred)

x_test = season_test_gen
for idx, class_idx in enumerate(class_pred):
    x_temp = x_test[int(idx/32)][0][idx%32] * 255
    x_temp = x_temp.astype('uint8')
    cv2.imwrite('prediction/{0}/{1:04d}.jpg'.format(class_names[class_idx], idx), x_temp[..., ::-1])
x_test[int(4/32)][0].shape
len(x_test)
len(class_pred)

plt.imshow(x_temp)
