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
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

'''
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
class_names = [x.split('/')[-1] for x in glob('smalldataset/train/*')] # クラス名をとってくる
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

# x_paths, y_classes = load_img_path_and_class('dataset/')


# split train and test
# train_paths, test_paths, train_classes, test_classes, train_likes, test_likes = train_test_split(x_paths, y_classes, y_likes, test_size=0.1, random_state=1225)

# input data profile
season_classes = len(class_names) # 4 season
rank_classes = np.max(y_likes)+1
# input_shape = len(train_paths)
# print(input_shape)
print(season_classes)
print(rank_classes)


# 画像枚数が多いのでgeneratorで渡す
# 季節分類
season_train_gen = MyGenerator(train_paths, train_classes, num_of_class=season_classes, batch_size=128)
season_test_gen = MyGenerator(test_paths, test_classes, num_of_class=season_classes, batch_size=128)
# お気に入り数分類
rank_train_gen = MyGenerator(train_paths, train_likes, num_of_class=rank_classes, batch_size=128)
rank_test_gen = MyGenerator(test_paths, test_likes, num_of_class=rank_classes, batch_size=128)
'''


# get class name
# class_names = [x.split('/')[-1] for x in glob('google_dataset/train/*')] # クラス名をとってくる
# print(class_names)

# input data profile
season_classes = 4 # 4 season

"""ImageDataGenerator"""

train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

season_train_gen = train_datagen.flow_from_directory(
        'broken_dataset_like300over_is_disappear/train',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

season_test_gen = test_datagen.flow_from_directory(
        'broken_dataset_like300over_is_disappear/test',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical',
        shuffle=False)


"""BASE MODEL"""
# create the base pre-trained model
# base_model = InceptionV3(weights='imagenet', include_top=False)
# base_model = VGG16(weights='imagenet', include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False)

# base_model.summary()
# len(base_model.layers)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Flatten()(x)
# classification
x_cls = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x_cls = Dropout(0.25)(x_cls)
x_cls = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x_cls)
x_cls = Dropout(0.5)(x_cls)
classification = Dense(season_classes, activation='softmax', name='classification')(x_cls)

## 季節分類
# this is the model we will train
# del model_cls
model_cls = Model(inputs=base_model.input, outputs=classification)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# train the model on the new data for a few epochs
history_cls = model_cls.fit_generator(season_train_gen,
                steps_per_epoch=len(season_train_gen),
                validation_data=season_test_gen,
                validation_steps=len(season_test_gen),
                epochs=3,
                shuffle=True
                )
model_cls.save_weights('ResNetaug_seasons_weight_base2.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# model_cls = load_model('seasons_VGGaug_weights_add_train.h5')
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
   # print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
# InceptionV3 :249
# ResNet50 :90
# VGG :15
for layer in model_cls.layers[:90]:
    layer.trainable = False
for layer in model_cls.layers[90:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model_cls.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
callbacks = []
# callbacks.append(EarlyStopping(patience=20))
callbacks.append(ModelCheckpoint(filepath='seasons_ResNetaug_weights_add_train.h5', save_best_only=True, save_weights_only=False))
history_cls_add = model_cls.fit_generator(season_train_gen,
                steps_per_epoch=len(season_train_gen),
                validation_data=season_test_gen,
                validation_steps=len(season_test_gen),
                epochs=100,
                callbacks=callbacks,
                shuffle=True
                )

# save and eval model
# evaluate(model_cls, x_test, cls_test, class_names)
# save_show_results(history, model)
# json_model_cls = model_cls.to_json()
# open('seasons_model.json', 'w').write(json_model_cls)
# model_cls.save_weights('seasons_small_weight_add.h5')
# show_history(history_cls)
acc = history_cls.history['acc'] + history_cls_add.history['acc']
val_acc = history_cls.history['val_acc'] + history_cls_add.history['val_acc']
# acc = history_cls_add.history['acc']
# val_acc = history_cls_add.history['val_acc']
plt.plot(range(1, len(acc)+1), acc, label='acc')
plt.plot(range(1, len(val_acc)+1), val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('ResNet_season_aug_classifier.png')
plt.show()


# prediction
# from keras.models import model_from_json
# del model_cls
# model_cls = load_model('google_seasons_VGGaug_weights_add_train.h5')
# json_string = open('seasons_model.json').read()
# model_cls = model_from_json(json_string)
# model_cls.load_weights('seasons_weight_add.h5')

datagen = ImageDataGenerator(rescale=1./255)

season_test_gen = datagen.flow_from_directory(
        'google_dataset/test',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical',
        shuffle=False)
print(season_test_gen.class_indices)

# どの画像がどのクラスへ分類されたかを保存
y_pred = model_cls.predict_generator(season_test_gen, steps=len(season_test_gen))
print(y_pred)

class_pred = np.argmax(y_pred, axis=1)
print(class_pred)
import collections
print(collections.Counter(class_pred))

x_test = season_test_gen
class_names = dict([(v,k) for k,v in x_test.class_indices.items()])
print(class_names)
for idx, predict in enumerate(class_pred):
    x_temp = x_test[int(idx/32)][0][idx%32] * 255
    x_temp = x_temp.astype('uint8')
    y_temp = x_test[int(idx/32)][1][idx%32]
    y_temp = np.argmax(y_temp)
    cv2.imwrite('prediction/{pred}/{ture}_{idx:04d}.jpg'.format(pred=class_names[predict], ture=class_names[y_temp], idx=idx), x_temp[..., ::-1])
