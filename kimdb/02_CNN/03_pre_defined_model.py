import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

vgg_model = vgg16.VGG16(weights='imagenet')

filename = 'pic.jpg'
org = load_img(filename, target_size=(224,224))
img = img_to_array(org)
plt.imshow(np.uint8(img))
plt.show()

# print encoding status of prediction
x = np.expand_dims(img, axis=0)
x = vgg16.preprocess_input(x)
pred = vgg_model.predict(x)
print(pred)

# decode
from keras.applications.imagenet_utils import decode_predictions
pred = decode_predictions(pred)
print(pred)
# prediction : 'fox_squirrel', 0.6382825 (여우다람쥐)

vgg_model.summary()

# transfer learning - Fine tuning
# 1. layer -> not trainable -> param should be replaced 0
for layer in vgg_model.layers:
    layer.trainable = False
vgg_model.summary()

# 2. bottleneck feature

