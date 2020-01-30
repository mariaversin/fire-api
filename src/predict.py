import json
import keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import keras
from keras import backend as k
from keras.models import Sequential, load_model,model_from_json
from keras.preprocessing.image import img_to_array
from flask import Flask, request
import os
import subprocess

def load_model():
    with open ('Model_0.97_150x150.json','r+') as f:
        model_json = json.load(f)
    global model
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('Model_0.97_150x150.h5')


def preprocess_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(im,(150,150))
    print(img.shape)
    img = np.array(img)
    print(img.shape)
    img = np.expand_dims(img,axis=0)
    print(img.shape)
    return img

def get_prediction(path):
    prediction = model.predict(preprocess_image(path))[0]
    subprocess.call(['speech-dispatcher'])  
    clases = ['Fire', 'Neutral']
    for i, p in enumerate(prediction):
        if p > 0.5:
            dicc =  {
                "classes":clases,
                "prediction": {
                    "label":clases[i],
                    "prob": str(p)}}
            if 'Fire' in dicc['prediction']['label']:
                subprocess.call(['spd-say', '"BE CAREFUL, FIRE!"'])
            else:
                subprocess.call(['spd-say', '"ALL RIGHT!"'])
    return dicc if clases == 'Neutral' else dicc
    

    
