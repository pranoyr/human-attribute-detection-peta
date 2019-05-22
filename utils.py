import os 
import numpy as np


def create_dict_from_labels(label_file):
    """ Create dictionary from Labes.txt
    """
    dict_labels = {}
    for i in label_file.split('\n'):
        if i:
            text = i.split()
            label_text = ' '.join(text[1:])
            dict_labels.update({text[0]:label_text})
    return dict_labels


def get_imgs_and_labels(dataset_path):
    """ Loading the img_ids and labels.
    """
    img_ids = []
    labels = []

    for folder in os.listdir(dataset_path):
        # creating dictionary from Label.txt
        label_file = open(os.path.join(dataset_path, folder, 'Label.txt'),'r').read()
        dict_labels = create_dict_from_labels(label_file)
        # creating img_ids and labels
        for filename in os.listdir(os.path.join(dataset_path, folder)):
            final_path = os.path.join(dataset_path, folder, filename)
            if final_path.split('.')[1]!='txt':
                img_ids.append(final_path)
                id = filename.split('_')[0]
                label = dict_labels[id]
                labels.append(label.lower())
        
    return img_ids, labels

def pre_process(img_batch):
    """ Input data preprocessing.
    """
    img_batch = img_batch.astype('float32')
    img_batch = img_batch / 255.0
    return img_batch

def one_hot_encode(integer_encodings, num_classes):
    """ One hot encode for multi-label classification.
    """
    onehot_encoded = []
    for integer_encoded in integer_encodings:
        letter = [0 for _ in range(num_classes)]
        for value in integer_encoded:
            letter[value] = 1
        onehot_encoded.append(letter)

    onehot_encoded = np.array(onehot_encoded)
    return onehot_encoded