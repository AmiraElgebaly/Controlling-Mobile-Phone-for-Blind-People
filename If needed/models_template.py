import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm



DATA_PATH = "data/train-Commands/"

#DATA_PATH2 = "test-numbers"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)

    label_indices = np.arange(0, len(labels)) # 0 size of classes
    labels = label_indices.astype(str)
    print(labels)
    print(label_indices)
    #give each class an label
    return labels, label_indices, to_categorical(label_indices)


#########################################################################################
# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=48):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=60)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

######################################################################################
def save_data_to_array(path=DATA_PATH, max_len=48):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.8, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)
    print("labels shape ", labels[0].shape)
    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    print("X shape ", X.shape)
    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)

# def get_test():
#     # Get available labels
#     labels, indices, _ = get_labels(DATA_PATH2)
#
#     # Getting first arrays
#     X = np.load(labels[0] + '.npy')
#     y = np.zeros(X.shape[0])
#
#     # Append all of the dataset into one single array, same goes for y
#     for i, label in enumerate(labels[1:]):
#         x = np.load(label + '.npy')
#         X = np.vstack((X, x))
#         y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))
#
#     assert X.shape[0] == len(y)
#     return X, y
#




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
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


# def load_dataset(path=DATA_PATH):
#     data = prepare_dataset(path)
#
#     dataset = []
#
#     for key in data:
#         for mfcc in data[key]['mfcc']:
#             dataset.append((key, mfcc))
#
#     return dataset[:100]

#########################################################################
save_data_to_array(DATA_PATH)

import keras
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

X_train, X_test, y_train, y_test = get_train_test()
'''#X_train, X_test, y_train, y_test = load_dataset(DATA_PATH)
print("X_train", "y_train")
print(X_train, y_train)
print("X_test", "y_test")
print(X_test, y_test)'''
#X_Test, Y_Test = get_test()
#Y_Test = Y_Test.reshape(Y_Test.shape[0], 1)
#print("X_Test", "Y_Test")
#print(X_Test.shape, Y_Test.shape)
#print("**************************************************")
X_train = X_train.reshape(X_train.shape[0], 60, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 60, 48, 1)
#X_Test = X_Test.reshape(X_Test.shape[0], 20, 11, 1)
print("X_train", "y_train_hot")
y_train_hot = to_categorical(y_train)
print(X_train.shape, y_train_hot.shape)

print("X_test", "y_test_hot")
y_test_hot = to_categorical(y_test)
print(X_test.shape, y_test_hot.shape)

#print("X_Test", "Y_Test_hot")
#Y_Test_hot = to_categorical(Y_Test)
#print(X_Test.shape, Y_Test_hot.shape)




listx = []
listy = []
# Getting the MFCC
print("wwwwwwwwwwwwwwwwwwww")
for i in range(0,len(os.listdir("test-Commands"))):

 for j in os.listdir('test-Commands/'+str(i)):

    sample = wav2mfcc('test-Commands/'+str(i)+"/"+j)# take full pass
    sample_reshaped = sample.reshape(1, 60, 48, 1)
    listx.append(sample_reshaped) # list of features
    listy.append(i)# list of actual

    #X = np.array(sample_reshaped)
    #Y = np.array(i)
X_Test = np.array(listx)
X_Test = X_Test.reshape(X_Test.shape[0], 60, 48, 1)
Y_Test = np.array(listy)
Y_Test_hot = to_categorical(Y_Test)
print("X_Test", "Y_Test_hot")
print(X_Test.shape, Y_Test_hot.shape)
#model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(60, 48, 1), pooling='max', classes=13)
model = VGG19(include_top=True, weights=None, input_tensor=None, input_shape=(60, 48, 1), pooling='max', classes=13)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])






history = model.fit(X_train, y_train_hot, batch_size=16, epochs=50, verbose=1, validation_data=(X_test, y_test_hot))
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

score = model.evaluate(X_Test, Y_Test_hot, verbose=0)

test_y_predictions = model.predict(X_Test)
print(" Y prediction : ", test_y_predictions.shape)
eva = np.zeros((2, Y_Test_hot.shape[1]))

for index in range(test_y_predictions.shape[0]):
    for c in range(Y_Test_hot.shape[1]):
        if (np.argmax(Y_Test_hot[index]) == c):
            if np.argmax(test_y_predictions[index]) == np.argmax(Y_Test_hot[index]):
                eva[0, c] = eva[0, c] + 1
            else:
                eva[1, c] = eva[1, c] + 1
print("acc : " ,score)
print("my evaluation : ",eva)
# print("history :  ",history)
##############################################################################################
# Plot training & validation accuracy values






