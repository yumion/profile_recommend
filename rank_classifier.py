from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import re
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras_preprocessing.image import img_to_array, list_pictures, load_img
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.objectives import categorical_crossentropy, mse
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.applications import inception_v3, resnet50, vgg16, vgg16
from keras import backend as K
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


dataset_dir = 'rank_dataset'

# input data profile
rank_classes = 3
batch_size = 128

train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            zoom_range=[0,1.2],
            horizontal_flip=True,
            fill_mode='constant')

test_gen = ImageDataGenerator(rescale=1./255)

rank_train_gen = train_gen.flow_from_directory(
                dataset_dir+'/train',
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False)

rank_test_gen = test_gen.flow_from_directory(
                dataset_dir+'/test',
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False)




"""BASE MODEL"""
# create the base pre-trained model
# base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
# base_model = resnet50.ResNet50(weights='imagenet', include_top=False)
base_model = vgg16.VGG16(weights='imagenet', include_top=False)

# base_model.summary()
# len(base_model.layers)
# for i, layer in enumerate(base_model.layers):
   # print(i, layer.name)


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# regression
x_rgs = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x_rgs = Dropout(0.25)(x_rgs)
x_rgs = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x_rgs)
x_rgs = Dropout(0.5)(x_rgs)
regression = Dense(rank_classes, activation='softmax', name='regression')(x_rgs)

## お気に入り数

# クラスに重み付け
ranks = [rank.split('/')[-2] for rank in glob(dataset_dir+'/train/*/*')]
num_likes = np.zeros(3)
for rank in ranks:
    rank_idx = rank_train_gen.class_indices[rank]
    num_likes[rank_idx] += 1
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

callbacks = []
callbacks.append(CSVLogger('VGG16_ranking_log1.csv', separator=',', append=True))
history_value = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=rank_test_gen,
                validation_steps=validation_steps,
                epochs=3,
                class_weight=class_weight,
                shuffle=True,
                callbacks=callbacks
                )
model_value.save_weights('VGG16_clsw_rank_weight_base.h5')
# InceptionV3 :249
# ResNet50 :90
# VGG16 :15
# VGG19 :17
for layer in model_value.layers[:15]:
    layer.trainable = False
for layer in model_value.layers[15:]:
    layer.trainable = True

from keras.optimizers import SGD
model_value.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

# callbacks.append(EarlyStopping(patience=20))
callbacks.append(ModelCheckpoint(filepath='favo_VGG_clsw_model_add_train.h5', save_best_only=True, save_weights_only=False))
history_value_add = model_value.fit_generator(rank_train_gen,
                steps_per_epoch=steps_per_epoch,
                validation_data=rank_test_gen,
                validation_steps=validation_steps,
                epochs=100,
                callbacks=callbacks,
                shuffle=True,
                class_weight=class_weight,
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
plt.legend()
plt.savefig('VGG16_ranking_classifier1.png')
plt.show()
# model_value.save('rank_model_add.h5')
