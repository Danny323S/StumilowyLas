import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv


# Ścieżka do zapisanego modelu
target_folder = 'saved models'
file_name = 'model'
extension = 'keras'
path = f'{target_folder}\\{file_name}.{extension}'

# Ładowanie modelu
model = load_model(path)
print(f'Model załadowany z: {path}')

img = cv.imread(rf"C:\Users\niepo\OneDrive\Obrazy\dab_moje\1.jpg")
print("Img shape: " +str(img.shape))
print(type(img))
predictions = model.predict(np.array(img,dtype='uint8'))
print("Predictions " + str(predictions))