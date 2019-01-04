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
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras_preprocessing.image import img_to_array, list_pictures, load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, Add, Activation

from ncc.preprocessing import preprocess_input, get_dataset
from ncc.validations import save_show_results, evaluate


## get x,y from dataset
def img_load_from_dir(target_dir):
    x_array, y_array = [], []
    file_list = glob(target_dir + '/*/')
    for class_index, folder_name in enumerate(file_list):
        for picture in list_pictures(folder_name):
            img = load_img(picture) # img type = PIL.image
            img_array = img_to_array(img) # np.array
            x_array.append(img_array.astype('uint8')) # input image
            y_array.append(class_index) # label
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    return x_array, y_array

# image load
x_array, y_array = img_load_from_dir('dataset')

# get class name
class_names = [x.split('/')[-1] for x in glob('dataset/*')] # クラス名をとってくる
print(class_names)

# train split
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.1, random_state=1225)

# image sizeを統一(predictのところで元のサイズで再読み込みしたいのでここで行う)
def resize_img_array(img_array, height=416, width=416):
    temp_array = []
    for i in range(len(img_array)):
        temp_array.append(cv2.resize(img_array[i], (height, width)))
    img_array = np.asarray(temp_array)
    return img_array

x_train = resize_img_array(x_train)
x_test = resize_img_array(x_test)

# preprocessing
x_train, y_train = preprocess_input(x_train, y_train)
x_test, y_test = preprocess_input(x_test, y_test)

# x_train[0].shape
# plt.imshow(x_train[0])
# plt.show()

# input data profile
num_classes = len(class_names) # 4 season
input_shape = x_train.shape[1:]
print(input_shape)

# model
'''
def resblock(x, filters, kernel_size=(3,3)):
    x_ = Conv2D(filters, kernel_size, padding='same')(x)
    x_ = BatchNormalization()(x_)
    x_ = Conv2D(filters, kernel_size, padding='same')(x_)
    x = Add()([x_, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

inputs = Input(shape=input_shape, name='input')

x = Conv2D(32, kernel_size=(5,5), activation='relu', padding='same')(inputs)
x = resblock(x, 32, kernel_size=(5,5))
x = resblock(x, 32, kernel_size=(5,5))
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(x)
x = resblock(x, 64, kernel_size=(5,5))
x = resblock(x, 64, kernel_size=(5,5))
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same')(x)
x = resblock(x, 128, kernel_size=(5,5))
x = resblock(x, 128, kernel_size=(5,5))
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(256, kernel_size=(5,5), activation='relu', padding='same')(x)
x = resblock(x, 256, kernel_size=(5,5))
x = resblock(x, 256, kernel_size=(5,5))
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='prediction')(x)
'''
inputs = Input(shape=input_shape, name='input')
x = Conv2D(32, kernel_size=(5,5), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Conv2D(256, kernel_size=(5,5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', name='prediction')(x)

model = Model(inputs=inputs, outputs=predictions)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
callbacks = [EarlyStopping(patience=10)]

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test)
                    )

# save and eval model
evaluate(model, x_test, y_test, class_names)

# save_show_results(history, model)
model.save('seasons_model2.h5')

del model
model = load_model('seasons_model.h5')


# どの画像がどのクラスへ分類されたかを保存
y_pred = model.predict(x_test)
# print(y_pred)
class_pred = np.argmax(y_pred, axis=1)
# print(class_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.1, random_state=1225) #再読み込み
for idx, class_idx in enumerate(class_pred):
    cv2.imwrite('prediction/{0}/{1:03d}.jpg'.format(class_names[class_idx], idx), x_test[idx])

# plt.imshow(x_test[0])
