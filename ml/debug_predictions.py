import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_path = 'models/visual_balance.h5'
model = load_model(model_path)
print(f"Loaded {model_path}")

dataset_dir = '../dataset/visual_balance'
classes = ['class_0', 'class_1']

def predict_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    pred = model.predict(img_array, verbose=0)
    return pred[0][0]

for cls in classes:
    files = glob.glob(os.path.join(dataset_dir, cls, '*.png'))[:5]
    print(f"\n--- Testing {cls} ---")
    for f in files:
        conf = predict_img(f)
        print(f"{os.path.basename(f)}: {conf:.4f} -> {'class_1' if conf >= 0.5 else 'class_0'}")
