from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
import tensorflow as tf

categories={0:'arduino_nano',1: 'arduino_uno',2: 'msp430', 3:'raspberry_pi'}


model = tf.keras.models.load_model('all_boards_model2.h5')

img_path='dataset_check/test/raspberry_pi/Raspberry PI_12.png'
img=image.load_img(img_path,target_size=(224, 224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
pred=model.predict(x)
y_pred=np.argmax(pred,axis=1)
print(y_pred)
print(type(y_pred))
print('predicted baord: ',categories[y_pred[0]])