from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import re
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras_preprocessing.image import img_to_array, list_pictures, load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.objectives import categorical_crossentropy, mse
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.applications import inception_v3, resnet50, vgg19
from keras import backend as K
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


dataset_dir = 'raw_dataset'


# get class name
class_names = [x.split('/')[-1] for x in glob(dataset_dir+'/train/*')] # クラス名をとってくる
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
    regex = re.compile(r'like(.*)_views(.*).jpg')
    for picture in pictures:
        # ラベル y
        mo = regex.search(picture) # ファイル名からお気に入り数を取得
        num_likes = mo.group(1)
        y_array.append(int(num_likes)) # label
    return y_array

# お気に入り数をランク分けに変換
def convert_to_class(y_likes):
    for idx, likes in enumerate(y_likes):
        if likes > 200:
            y_likes[idx] = 2
        elif likes > 100 and likes <= 200:
            y_likes[idx] = 1
        elif likes <= 100:
            y_likes[idx] = 0
    return y_likes


y_likes = load_num_likes_from_dir(dataset_dir+'/*/*')
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
plt.savefig('distribution_favorite_divideTo3.png')
plt.show()


# input data profile
rank_classes = np.max(y_likes)+1
# input_shape = len(train_paths)
# print(input_shape)
print(rank_classes)


def rankGenerator(target_dir, num_classes=3, subset=None, split_ratio=0.1, batch_size=32, height=224, width=224):
    file_list = glob(target_dir+'*/*')
    regex = re.compile(r'like(.*).jpg')
    # train test split
    train_size = len(file_list) * (1-split_ratio)
    train_size = int(train_size)
    if subset is None:
        split_file_list = file_list
    else:
        np.random.shuffle(file_list)
        if subset == 'training':
            split_file_list = file_list[:train_size]
        elif subset == 'validation':
            split_file_list = file_list[train_size:]

    while True:
        x_array, y_array = [], []
        for img_path in split_file_list:
            # image
            img = load_img(img_path) # img type = PIL.image
            img_array = img_to_array(img) # np.array
            img_array = cv2.resize(img_array, (height, width))
            x_array.append(img_array.astype('float32'))
            # ラベル y
            mo = regex.search(img_path) # ファイル名からお気に入り数を取得
            num_likes = mo.group(1)
            y_array.append(int(num_likes))
            if len(x_array) == batch_size:
                # 入力x
                x_array = np.asarray(x_array)
                x_array /= 255.
                # 教師ラベルy
                y_array = convert_to_class(y_array)
                y_array = np.asarray(y_array)
                y_array = to_categorical(y_array, num_classes=num_classes) #one-hot
                yield x_array, y_array
                x_array, y_array = [], []

batch_size = 128
rank_train_gen = rankGenerator(dataset_dir+'/train/', 3, batch_size=batch_size)
rank_test_gen = rankGenerator(dataset_dir+'/test/', 3, batch_size=batch_size)




"""BASE MODEL"""
# create the base pre-trained model
# base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
# base_model = resnet50.ResNet50(weights='imagenet', include_top=False)
base_model = vgg19.VGG19(weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# regression
x_rgs = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x_rgs = Dropout(0.25)(x_rgs)
x_rgs = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x_rgs)
x_rgs = Dropout(0.5)(x_rgs)
regression = Dense(rank_classes, activation='softmax', name='regression')(x_rgs)

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

# FC層のみを学習
for layer in base_model.layers:
    layer.trainable = False

model_value.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# train the model on the new data for a few epochs
steps_per_epoch = int(len(glob(dataset_dir+'/train/*/*')) // batch_size)
validation_steps = int(len(glob(dataset_dir+'/test/*/*')) // batch_size)
history_value = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=rank_test_gen,
                validation_steps=validation_steps,
                epochs=3,
                shuffle=True
                )
model_value.save_weights('VGG19_notclsw_favo_weight_base.h5')
# InceptionV3 :249
# ResNet50 :90
# VGG16 :15
for layer in model_value.layers[:90]:
    layer.trainable = False
for layer in model_value.layers[90:]:
    layer.trainable = True

from keras.optimizers import SGD
model_value.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
callbacks = []
# callbacks.append(EarlyStopping(patience=20))
callbacks.append(ModelCheckpoint(filepath='favo_ResNet_notclsw_model_add_train.h5', save_best_only=True, save_weights_only=False))
history_value_add = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=rank_test_gen,
                validation_steps=validation_steps,
                epochs=100,
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True
                )

'''
for layer in model_value.layers[:90]:
    layer.trainable = True

model_value.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
history_value_add2 = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=rank_test_gen,
                validation_steps=validation_steps,
                epochs=100,
                class_weight=class_weight,
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=True
                )
'''
# save and eval model
# evaluate(model_value, x_test, value_test)
# show_history(history_value)
acc = history_value.history['acc'] + history_value_add.history['acc']# + history_value_add2.history['acc']
val_acc = history_value.history['val_acc'] + history_value_add.history['val_acc']# + history_value_add2.history['val_acc']
plt.plot(range(len(acc)), acc, label='acc')
plt.plot(range(len(val_acc)), val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('favo_classifier3.png')
plt.show()
# model_value.save('rank_model_add.h5')
