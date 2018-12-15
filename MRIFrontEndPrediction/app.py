from flask import Flask, render_template,  flash, redirect, url_for, session, request, make_response
import random,os, csv
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'E:\digital_transformation_source_code\static\images'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'tiff'])




def get_model():
    global model
    model = load_model('E:\MySavedModels\custom_model.h5')
    model._make_predict_function()
    global graph
    graph = tf.get_default_graph()

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    print("**** Model Loaded ***")



def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print("*.. Loading Keras Model")
get_model()



app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict', methods=["GET", "POST"])
def predictImage():
    if request.method == 'POST':
        # check if the post request has the file part

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = load_img(full_path, target_size=(150, 150))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            with graph.as_default():
                prediction = model.predict(x, batch_size=10)
                print (prediction)
            #return redirect(url_for('uploaded_file',filename=filename))


    return render_template("predict.html", **locals())




if(__name__ == '__main__'):
    app.run(debug=True)