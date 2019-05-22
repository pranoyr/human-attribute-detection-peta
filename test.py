
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
    names_to_labels.update({label:i})


def one_hot_encode(integer_encodings, num_classes):
    """ One hot encode for multi-label classification.
    """
    onehot_encoded = []
    for integer_encoded in integer_encodings:
        letter = [0 for _ in range(num_classes)]
        for value in integer_encoded:
            letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


# for anns in label_filename.split('\n'):
#     dict_labels.update(label_filename.split('\n')[9])

def create_dict_from_labels():
    # create dict from Labes.txt
    file = open("/Users/pranoyr/PycharmProjects/human-attribute-classification-peta/PETA/3DPeS/Label.txt","r").read()
    dict_labels = {}
    for i in file.split('\n'):
        if i:
            text = i.split()
            label_text = ' '.join(text[1:])
            dict_labels.update({text[0]:label_text})
    return dict_labels

from PIL import Image
import numpy as np
import cv2
img = Image.open('/Users/pranoyr/Desktop/sample.jpg')
img = img.resize((100, 200))
img = np.array(img)

cv2.imshow('window',img)
cv2.waitKey(0)