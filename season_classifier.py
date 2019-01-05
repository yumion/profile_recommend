'''画像ファイル名を統一
import glob
import os

for season in ['spring', 'summer', 'autumn', 'winter']:
    i = 0
    img_list = []
    for image_suffix in ['jpg', 'png', 'jpeg', 'JPG', 'PNG']:
        image_paths = 'dataset/'+season+'/*.' + image_suffix
        img_list += glob.glob(image_paths)
    for file in img_list:
        suffix = file.split('.')[-1]
        os.rename(file, 'dataset/' + season + '/%03d.'%(i) + suffix)
        i += 1
'''

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
from keras.preprocessing import image
from keras import backend as K

from ncc.preprocessing import preprocess_input
from ncc.validations import save_show_results, evaluate
from ncc.models import Model2D

## get x,y from dataset
def img_load_from_dir(target_dir):
    x_array, y_array = [], []
    file_list = glob(target_dir + '/*/')
    regex = re.compile(r'like(.*).jpg')
    for class_index, folder_name in enumerate(file_list):
        for picture in list_pictures(folder_name):
            # 入力 x
            img = load_img(picture) # img type = PIL.image
            img_array = img_to_array(img) # np.array
            x_array.append(img_array.astype('uint8')) # input image
            # ラベル y
            mo = regex.search(picture) # ファイル名からお気に入り数を取得
            num_likes = mo.group(1)
            y_array.append([class_index, int(num_likes)]) # label
    # ndarrayに変換
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    return x_array, y_array

# image load
x_array, y_array = img_load_from_dir('dataset')

# get class name
class_names = [x.split('/')[-1] for x in glob('dataset/*')] # クラス名をとってくる
print(class_names)

# お気に入り数の分布
np.min(y_array[:,1])
num_likes = np.zeros(1200)
for like in y_array[:,1]:
    num_likes[like] += 1
plt.bar(np.arange(1200), num_likes)
plt.xlim(100, 1160)
plt.show()

# train split
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.1, random_state=1225)

# image sizeを統一(predictのところで元のサイズで再読み込みしたいのでここで行う)
def resize_img_array(img_array, height=299, width=299):
    temp_array = []
    for i in range(len(img_array)):
        temp_array.append(cv2.resize(img_array[i], (height, width)))
    img_array = np.asarray(temp_array)
    return img_array

x_train = resize_img_array(x_train)
x_test = resize_img_array(x_test)

# preprocessing
x_train, cls_train = preprocess_input(x_train, y_train[:,0])
x_test, cls_test = preprocess_input(x_test, y_test[:,0])
# クラスラベルだけone-hotにして、お気に入り数を右に追加
value_train = np.expand_dims(y_train[:,1], axis=-1)
value_test = np.expand_dims(y_test[:,1], axis=-1)
# クラスラベルと回帰値をドッキングした教師ラベル
# y_train_combined = np.hstack([cls_train, value_train])
# y_test_combined = np.hstack([cls_test, value_test])

# cls_train.shape
# value_train.shape
# y_train_combined.shape
# plt.imshow(x_train[0])
# plt.show()

# input data profile
num_classes = len(class_names) # 4 season
input_shape = x_train.shape[1:]
print(input_shape)

'''
# model
inputs = Input(shape=input_shape, name='input')
# feature extraction
x = Conv2D(32, kernel_size=(5,5), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Conv2D(32, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)
x = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)
x = Conv2D(256, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

# classification
x_cls = Flatten()(x)
x_cls = Dense(512, activation='relu')(x_cls)
x_cls = Dropout(0.5)(x_cls)
classification = Dense(num_classes, activation='softmax', name='classification')(x_cls)

# regression
x_rgs = Flatten()(x)
x_rgs = Dense(1024, activation='relu')(x_rgs)
x_rgs = Dropout(0.25)(x_rgs)
x_rgs = Dense(256, activation='relu')(x_rgs)
x_rgs = Dropout(0.5)(x_rgs)
regression = Dense(1, activation='relu', name='regression')(x_rgs)
'''

'''
# 同時学習
model = Model(inputs, [classification, regression])
model.summary()
# plot_model(model, show_shapes=True)
model.compile(optimizer=SGD(momentum=0.9, nesterov=True), loss={'classification': 'categorical_crossentropy', 'regression': 'mse'}, loss_weights={'classification': 0.95, 'regression': 0.05}, metrics=['acc'])

callbacks = [EarlyStopping(patience=5)]
history = model.fit(x_train, [cls_train, value_train],
                    epochs=20,
                    batch_size=32,
                    callbacks=callbacks,
                    validation_data=(x_test, [cls_test, value_test])
                    )



# save_show_results(history, model)
model.save('classifierAndRegression_model.h5')

# load_model
del model
model = load_model('classifierAndRegression_model.h5')
'''

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# classification
x_cls = Dense(1024, activation='relu')(x)
classification = Dense(num_classes, activation='softmax', name='classification')(x_cls)

# regression
x_rgs = Dense(1024, activation='relu')(x)
regression = Dense(5, activation='relu', name='regression')(x_rgs)

## 季節分類
# this is the model we will train
model_cls = Model(inputs=base_model.input, outputs=classification)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model_cls.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# train the model on the new data for a few epochs
callbacks = [EarlyStopping(patience=5)]
history_cls = model_cls.fit(x_train, cls_train,
                            epochs=3,
                            batch_size=32,
                            callbacks=callbacks,
                            validation_data=(x_test, cls_test)
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
history_cls = model_cls.fit(x_train, cls_train,
                            epochs=1,
                            batch_size=32,
                            callbacks=callbacks,
                            validation_data=(x_test, cls_test)
                            )

## お気に入り数の回帰
model_value = Model(inputs=base_model.input, outputs=regression)

for layer in base_model.layers:
    layer.trainable = False

model_value.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# train the model on the new data for a few epochs
history_value = model_value.fit(x_train, value_train,
                            epochs=3,
                            batch_size=32,
                            validation_data=(x_test, value_test)
                            )

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model_value.layers[:249]:
    layer.trainable = False
for layer in model_value.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model_value.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history_value = model_value.fit(x_train, value_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(x_test, value_test)
                            )

'''
# 別々に学習
model_cls = Model(inputs=inputs, outputs=classification)
# model_cls.summary()
model_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model_rgs = Model(inputs=inputs, outputs=regression)
# model_rgs.summary()
model_rgs.compile(optimizer='adam', loss='mse', metrics=['acc'])

callbacks = [EarlyStopping(patience=5)]
history_cls = model_cls.fit(x_train, cls_train,
                            epochs=20,
                            batch_size=32,
                            callbacks=callbacks,
                            validation_data=(x_test, cls_test)
                            )

history_rgs = model_rgs.fit(x_train, value_train,
                            epochs=20,
                            batch_size=32,
                            callbacks=callbacks,
                            validation_data=(x_test, value_test)
                            )
'''

# save and eval model
evaluate(model_cls, x_test, y_test, class_names)

# save_show_results(history, model)
model_cls.save('seasons_model.h5')

del model_cls
model_cls = load_model('seasons_model.h5')


# どの画像がどのクラスへ分類されたかを保存
y_pred = model.predict(x_test)
# print(y_pred)
class_pred = np.argmax(y_pred, axis=1)
# print(class_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.1, random_state=1225) #リサイズしているため再読み込み
for idx, class_idx in enumerate(class_pred):
    cv2.imwrite('prediction/{0}/{1:04d}.jpg'.format(class_names[class_idx], idx), x_test[idx])

# plt.imshow(x_test[0])
