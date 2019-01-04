from PIL import Image
import numpy as np

from yolo import YOLO


'''
## 自分の手持ち画像で分類する
X = cv2.imread('dataset/spring/013.jpg')
X = cv2.resize(X, (416, 416))
X = np.expand_dims(X, axis=0)
print(X.shape)
'''

img = 'dataset/spring/013.jpg'
image = Image.open(img)
image.shape

r_image = YOLO.detect_image(image)
r_image.show()
