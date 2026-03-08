import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model = load_model('models/visual_balance.h5')

def predict_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    pred = model.predict(img_array, verbose=0)
    return pred[0][0]

for cls in ['class_0', 'class_1']:
    files = glob.glob(f'../dataset/visual_balance/{cls}/*.png')
    c0 = 0
    c1 = 0
    for f in files:
        if predict_img(f) >= 0.5:
            c1 += 1
        else:
            c0 += 1
    print(f"{cls} true label -> predicted class_0: {c0}, class_1: {c1}")
