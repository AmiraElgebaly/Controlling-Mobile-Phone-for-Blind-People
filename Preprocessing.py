import librosa
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, TimeDistributed, Conv1D, ZeroPadding1D, GRU
from keras.layers import Lambda, Input, Dropout, Masking, BatchNormalization, Activation
from keras.models import Model, load_model
import matplotlib.pyplot as plt


DATA_PATH1 = "data/train-answers/"
DATA_PATH2 = "data/train-Commands/"
DATA_PATH3 = "data/train-numbers/"


# get command from google assistant

# vc.get_command()
# Folder_path = 'record.wav'
# vc.segmentation(Folder_path, minSilenceTime=0.35)

#region Get_Labels
"""
Input: Folder Path (ex. data/train_answers)
Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
"""

def get_labels(path):
    labels = os.listdir(path)

    label_indices = np.arange(0, len(labels))  # 0 size of classes
    labels = label_indices.astype(str)
    print(labels)
    print(label_indices)
    # give each class an label
    return labels, label_indices, to_categorical(label_indices)

#endregion

#region MFCC
""" Handy function to convert wav2mfcc """ 
def wav2mfcc(file_path, max_len=22):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=52)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

#endregion

#region Dataset Preparation
"""
Function: extract mfcc features from records and save them in .npy array 
Details: Save records exists in data/train_[answers/commands/numbers]/{label} 
            in {label}.npy[1|2|3].npy array with dimension (#records, 52 ,22)

Parameters: 
    path: path of records (ex. data/train_answers)
    max_len: default = 22 - used to extract mfcc features from record with dimension (52,22)
    ext: should be (.npy1.npy for answers) and (.npy2.npy for commands) and (.npy3.npy for numbers)
"""

def save_data_to_array(path, max_len=22, ext=''):
    labels, _, _ = get_labels(path) # labels = '0','1',...

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)] #list of all wavfiles
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len) # convert record to mfcc features with dimension (52,22)
            mfcc_vectors.append(mfcc)
        np.save(label + ext, mfcc_vectors) # save records' mfcc features in .npy with dimension (#records/label, 52 ,22)


"""
Function: out train_x, test_x, train_y, test_y that represent specific class [Numbers|Answers|Commands] data and labels for training purpose.
            (training and validation Data)
            merge all mfcc features of different labels for specific class into one array Called X 
            and their labels into other Array Called y -----> apply train_test_split() on them

Parameters: 
split_ratio: Trainning Ratio (default= 0.8)
random_state: random state for splitting (default= 42)
path: path of records (ex. data/train_answers) to get labels for specific class 
ext: extension of files that contain the mfcc features for records with different labels (ex. answers saved in extension {label}.npy1.npy) 
"""
def get_train_test(split_ratio=0.8, random_state=42, path='', ext=''):
    # Get available labels
    labels, indices, _ = get_labels(path) # labels= '0','1',... , indices= 0,1,...

    # Getting first arrays
    X = np.load(labels[0] + ext)
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + ext)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)

"""
Function: return X (MFCC features),y(labels) containing testing data and their labels

parameters: 
file_name: Folder that contains Label folders of testing records (ex.test_commands)
"""

def test_data(file_name):
    listx = []
    listy = []
    # Getting the MFCC
    print("wwwwwwwwwwwwwwwwwwww")
    for i in range(0, len(os.listdir(file_name))):

        # iterating over labels Folders i= 0,1,..
        for j in os.listdir(file_name + '/' + str(i)):
            sample = wav2mfcc(file_name + '/' + str(i) + "/" + j)  # take full pass
            # print(sample.shape)
            sample_reshaped = sample.reshape(1, 52, 22)
            listx.append(sample_reshaped)  # list of features
            listy.append(i)  # list of actual

            # X = np.array(sample_reshaped)
            # Y = np.array(i)
    X_Test = np.array(listx)
    X_Test = X_Test.reshape(X_Test.shape[0], 52, 22)
    Y_Test = np.array(listy)
    Y_Test_hot = to_categorical(Y_Test)
    print("X_Test", "Y_Test_hot")
    print(X_Test.shape, Y_Test_hot.shape)
    return X_Test, Y_Test_hot

#endregion

'''

def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=52)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]

'''
