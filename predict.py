import keras
from keras.models import load_model
from PIL import Image
import numpy as np
from utils import pre_process

import scipy.io

# creating names to labels and inverse mapping.
# creating names_to_labels mapping
names_to_labels = {}
all_labels = open("labels.txt").read()
all_labels = all_labels.split()

for color in ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']:
    all_labels.append('footwear'+color)
    all_labels.append('hair'+color)
    all_labels.append('lowerbody'+color)
    all_labels.append('upperbody'+color)
for i, label in enumerate(all_labels):
    names_to_labels.update({label.lower():i})
labels_to_names = {v: k for k, v in names_to_labels.items()}

# loading model
model = load_model('./snapshots/human_attribute_model_32.h5')
# load image
img_rgb  = Image.open('/Users/pranoyr/Desktop/test/sample4.jpg')
img_rgb = img_rgb.resize((150, 300))
img_batch = np.expand_dims(img_rgb, axis=0)
img_batch = pre_process(img_batch)
# prediction
results = model.predict(img_batch)
for i,score in enumerate(results[0]):
 if (score > 0.7):
    print(labels_to_names[i])