from flask import Flask,render_template,redirect,flash,url_for,request
import cv2
import numpy as np
import base64
import io
from PIL import Image
from werkzeug.utils import secure_filename
import os
UPLOAD_FOLDER = 'image_uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions

import tensorflow as tf


categories=['arduino_nano', 'arduino_uno', 'msp430', 'raspberry_pi']
model = tf.keras.models.load_model('all_boards_model2.h5')


@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    categories = {0: 'arduino_nano', 1: 'arduino_uno', 2: 'msp430', 3: 'raspberry_pi'}
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path=os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            if filename:
                print("yes")
            #image=cv2.imdecode(np.fromstring(request.files['file'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
            img=image.load_img(file_path,target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pred = model.predict(x)
            y_pred = np.argmax(pred, axis=1)
            result=categories[y_pred[0]]
            """rawbytes=io.BytesIO()
            img.save(rawbytes,'JPEG')
            rawbytes.seek(0)
            encoded_image=base64.b64encode(rawbytes.getvalue()).decode('ascii')
            mime='image/jpeg'
            uri="data:%s;base64,%s"%(mime,encoded_image)"""


            return result

    return render_template('index.html')