import numpy as np
import tensorflow as tf

from flask import Flask

app = Flask(__name__)

model = tf.keras.saving.load_model('models/mobilenet_fakeface')
img_path = 'fake/fake_625.jpg'

def prediction(image_path):

    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    output = model.predict(image)
    return output

@app.route('/')
def main():
    pred = prediction(img_path)
    if pred < 0.5 : return 'Fake'
    return '<h1>Real</h1>'