import os
import tensorflow as tf
from keras import models
import numpy as np
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_memory_growth(gpu, True)

def evaluate_prediction(p_val):
    if p_val > 0.5:
        print(f'Predicted as class: Dog')
    else:
        print(f'Predicted as class: Cat')
    return

def load_active_img(filename):
    img = cv2.imread(filename)
    resize = tf.image.resize(img, (256,256))
    return resize

model = models.load_model(os.path.join('model', 'cat-dog-classify_model.keras'))

filename = 'dogtest.jpg'
image = load_active_img(filename)
yhat = model.predict(np.expand_dims(image/255, 0))
evaluate_prediction(p_val=yhat)

filename = 'cattest.jpg'
image = load_active_img(filename)
yhat = model.predict(np.expand_dims(image/255, 0))
evaluate_prediction(p_val=yhat)
    
exit('Exited')