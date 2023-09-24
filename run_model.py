import os
import tensorflow as tf
from keras import models
import numpy as np
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

img = cv2.imread('dogtest.jpg')
resize = tf.image.resize(img, (256,256))

model = models.load_model(os.path.join('model', 'cat-dog-classify_model.h5'))
yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print(f'Predicted as a DOG')
else:
    print(f'Predicted as a CAT')