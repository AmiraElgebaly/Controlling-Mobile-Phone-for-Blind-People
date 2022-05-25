import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
#import female_preprocessing as fb
import matplotlib.pyplot as plt
import asr as vc
# print('Enter train folder name : ')
# file_train_name = input()
# DATA_PATH = "data/"+str(file_train_name)+"/"

DATA_PATH1 = "data/train-answers/"
DATA_PATH2 = "data/train-Commands/"
DATA_PATH3 = "data/train-numbers/"

# vc.get_command()
# Folder_path = 'record.wav'
# vc.segmentation(Folder_path, minSilenceTime=0.35)

# DATA_PATH2 = "test-Commands"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)

def get_labels(path):
    labels = os.listdir(path)

    label_indices = np.arange(0, len(labels))  # 0 size of classes
    labels = label_indices.astype(str)
    print(labels)
    print(label_indices)
    # give each class an label
    return labels, label_indices, to_categorical(label_indices)


#########################################################################################
# Handy function to convert wav2mfcc
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


######################################################################################
def save_data_to_array(path, max_len=22, ext=''):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + ext, mfcc_vectors)


def get_train_test(split_ratio=0.8, random_state=42, path='', ext=''):
    # Get available labels
    labels, indices, _ = get_labels(path)

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

#########################################################################


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, TimeDistributed, Conv1D, ZeroPadding1D, GRU
from keras.layers import Lambda, Input, Dropout, Masking, BatchNormalization, Activation
from keras.models import Model ,load_model

# save_data_to_array(path=DATA_PATH1, ext='.npy1')
# save_data_to_array(path=DATA_PATH2, ext='.npy2')
# save_data_to_array(path=DATA_PATH3, ext='.npy3')

X_train1, X_test1, y_train1, y_test1 = get_train_test(path=DATA_PATH1, ext='.npy1.npy')
X_train2, X_test2, y_train2, y_test2 = get_train_test(path=DATA_PATH2, ext='.npy2.npy')
X_train3, X_test3, y_train3, y_test3 = get_train_test(path=DATA_PATH3, ext='.npy3.npy')

'''#X_train, X_test, y_train, y_test = load_dataset(DATA_PATH)
print("X_train", "y_train")
print(X_train, y_train)
print("X_test", "y_test")
print(X_test, y_test)'''

# X_Test, Y_Test = get_test()
# Y_Test = Y_Test.reshape(Y_Test.shape[0], 1)
# print("X_Test", "Y_Test")
# print(X_Test.shape, Y_Test.shape)
# print("**************************************************")
X_train1 = X_train1.reshape(X_train1.shape[0], 52, 22)
X_test1 = X_test1.reshape(X_test1.shape[0], 52, 22)
# X_Test = X_Test.reshape(X_Test.shape[0], 20, 11, 1)
print("X_train", "y_train_hot")
y_train_hot1 = to_categorical(y_train1)
print(X_train1.shape, y_train_hot1.shape)

print("X_test", "y_test_hot")
y_test_hot1 = to_categorical(y_test1)
print(X_test1.shape, y_test_hot1.shape)

# print("X_Test", "Y_Test_hot")
# Y_Test_hot = to_categorical(Y_Test)
# print(X_Test.shape, Y_Test_hot.shape)

#########################################################################
# commands
X_train2 = X_train2.reshape(X_train2.shape[0], 52, 22)
X_test2 = X_test2.reshape(X_test2.shape[0], 52, 22)
# X_Test = X_Test.reshape(X_Test.shape[0], 20, 11, 1)
print("X_train", "y_train_hot")
y_train_hot2 = to_categorical(y_train2)
print(X_train2.shape, y_train_hot2.shape)

print("X_test", "y_test_hot")
y_test_hot2 = to_categorical(y_test2)
print(X_test2.shape, y_test_hot2.shape)

#########################################################################
# numbers

X_train3 = X_train3.reshape(X_train3.shape[0], 52, 22)
X_test3 = X_test3.reshape(X_test3.shape[0], 52, 22)
# X_Test = X_Test.reshape(X_Test.shape[0], 20, 11, 1)
print("X_train", "y_train_hot")
y_train_hot3 = to_categorical(y_train3)
print(X_train3.shape, y_train_hot3.shape)

print("X_test", "y_test_hot")
y_test_hot3 = to_categorical(y_test3)
print(X_test3.shape, y_test_hot3.shape)


def test_data(file_name):
    listx = []
    listy = []
    # Getting the MFCC
    print("wwwwwwwwwwwwwwwwwwww")
    for i in range(0, len(os.listdir(file_name))):

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


#############################################################################################
def test_asr(file_name):
    listx = []
    listy = []
    # Getting the MFCC
    print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
    for i in range(0, len(os.listdir(file_name))):

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


#############################################################################################



file_name1 = 'test-answers'
file_name2 = 'test-Commands'
file_name3 = 'test-numbers'
file_name4 = 'yes'


print('test answers')
X_Test1, Y_Test_hot1 = test_data(file_name1)
print('test commands')
X_Test2, Y_Test_hot2 = test_data(file_name2)
print('test numbers')
X_Test3, Y_Test_hot3 = test_data(file_name3)
print('test asr')
X_Test4, Y_Test_hot4 = test_data(file_name4)


import numpy as np
import tensorflow as tf
import tflearn
import tflearn.layers.merge_ops
#from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, reshape
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_1d, max_pool_1d, avg_pool_1d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
import pickle as cPickle
import tflearn.datasets.mnist as mnist
import os
import cv2
import numpy as np
import scipy.io.wavfile
import random


def cnn_lstm(input_dim, output_dim, dropout=0.2, n_layers=1):
    #     # Input data type
    dtype = 'float32'

    # ---- Network model ----
    input_data = Input(name='the_input', shape=input_dim, dtype=dtype)

    # 1 x 1D convolutional layers with strides 4
    x = Conv1D(filters=256, kernel_size=10, strides=4, name='conv_1')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout, name='dropout_1')(x)

    x = LSTM(128, activation='relu', return_sequences=True,
             dropout=dropout, name='lstm_1')(x)
    x = LSTM(128, activation='relu', return_sequences=False,
             dropout=dropout, name='lstm_2')(x)

    #     # 1 fully connected layer DNN ReLu with default 20% dropout
    x = Dense(units=64, activation='relu', name='fc')(x)
    x = Dropout(dropout, name='dropout_2')(x)

    # Output layer with softmax
    y_pred = Dense(units=output_dim, activation='softmax', name='softmax')(x)

    network_model = Model(inputs=input_data, outputs=y_pred)

    return network_model


input_dim = (52, 22)
class1 = 5
class2 = 13
class3 = 10

# K.clear_session()
model1 = cnn_lstm(input_dim, class1)
# K.clear_session()
model2 = cnn_lstm(input_dim, class2)
# K.clear_session()
model3 = cnn_lstm(input_dim, class3)

from keras.callbacks import TensorBoard

sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

# model1.compile(loss='categorical_crossentropy',
#                optimizer=adam,
#                metrics=['accuracy'])
# history1 = model1.fit(X_train1, y_train_hot1,
#                       batch_size=16, epochs=100,
#                       validation_data=(X_test1, y_test_hot1)
#                       )
# model1.save('my_model1.h5')
# print("model1 saved")
model1 = load_model('my_model1.h5')
model1.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])
##############################################################################################


# history = model.fit(X_train, y_train_hot, batch_size=16, epochs=120, verbose=1, validation_data=(X_test, y_test_hot))
# print(history1.history.keys())
# plt.plot(history1.history['acc'])
# plt.plot(history1.history['val_acc'])
# plt.title('Model accuracy *Answers*')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# plt.title('Model loss *Answers*')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

#score1 = model1.evaluate(X_Test4, Y_Test_hot4, verbose=0)
score1 = model1.predict(X_Test4)
test_y_predictions1 = model1.predict(X_Test4)

print(" Y prediction : ", test_y_predictions1.shape)
eva1 = np.zeros((2, Y_Test_hot1.shape[1]))

for index in range(test_y_predictions1.shape[0]):
    for c in range(Y_Test_hot1.shape[1]):
        if (np.argmax(Y_Test_hot1[index]) == c):
            if np.argmax(test_y_predictions1[index]) == np.argmax(Y_Test_hot1[index]):
                eva1[0, c] = eva1[0, c] + 1
            else:
                eva1[1, c] = eva1[1, c] + 1
print("acc 1 : ", score1)

# Perform forward pass
s1 = np.argmax(score1)
s1_prob = score1[0, s1]
print('s1 : ', s1)
print('prob1', s1_prob)
print("my evaluation 1 : ", eva1)
# print("history :  ",history)

##############################################################################################


###########################################################################################
# model2.compile(loss='categorical_crossentropy',
#                optimizer=adam,
#                metrics=['accuracy'])
# history2 = model2.fit(X_train2, y_train_hot2,
#                       batch_size=16, epochs=62,
#                       validation_data=(X_test2, y_test_hot2)
#                       )
# model2.save('my_model2.h5')
# print("model2 saved")

model2 = load_model('my_model2.h5')
model2.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

###############################################################################################

# commands
# print(history2.history.keys())
# plt.plot(history2.history['acc'])
# plt.plot(history2.history['val_acc'])
# plt.title('Model accuracy *Commands*')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history2.history['loss'])
# plt.plot(history2.history['val_loss'])
# plt.title('Model loss *Commands*')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

#score2 = model2.evaluate(X_Test4, Y_Test_hot4, verbose=0)
score2 = model2.predict(X_Test4)
test_y_predictions2 = model2.predict(X_Test4)

print(" Y prediction : ", test_y_predictions2.shape)
eva2 = np.zeros((2, Y_Test_hot2.shape[1]))

for index in range(test_y_predictions2.shape[0]):
    for c in range(Y_Test_hot2.shape[1]):
        if (np.argmax(Y_Test_hot2[index]) == c):
            if np.argmax(test_y_predictions2[index]) == np.argmax(Y_Test_hot2[index]):
                eva2[0, c] = eva2[0, c] + 1
            else:
                eva2[1, c] = eva2[1, c] + 1
print("acc 2 : ", score2)
s2 = np.argmax(score2)
s2_prob = score2[0, s2]
print('prob2', s2_prob)
print("my evaluation 2 : ", eva2)
#############################################################################################


# model3.compile(loss='categorical_crossentropy',
#                optimizer=adam,
#                metrics=['accuracy'])
# history3 = model3.fit(X_train3, y_train_hot3,
#                       batch_size=16, epochs=62,
#                       validation_data=(X_test3, y_test_hot3)
#                       )
#
# model3.save('my_model3.h5')
# print("model3 saved")

model3 = load_model('my_model3.h5')
model3.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])
#########################################################################################

# numbers

# print(history3.history.keys())
# plt.plot(history3.history['acc'])
# plt.plot(history3.history['val_acc'])
# plt.title('Model accuracy *numbers*')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history3.history['loss'])
# plt.plot(history3.history['val_loss'])
# plt.title('Model loss *numbers*')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

#score3 = model3.evaluate(X_Test4, Y_Test_hot4, verbose=0)
score3 = model3.predict(X_Test4)
test_y_predictions3 = model3.predict(X_Test4)
print(" Y prediction3 : ", test_y_predictions3.shape)
eva3 = np.zeros((2, Y_Test_hot3.shape[1]))

for index in range(test_y_predictions3.shape[0]):
    for c in range(Y_Test_hot3.shape[1]):
        if (np.argmax(Y_Test_hot3[index]) == c):
            if np.argmax(test_y_predictions3[index]) == np.argmax(Y_Test_hot3[index]):
                eva3[0, c] = eva3[0, c] + 1
            else:
                eva3[1, c] = eva3[1, c] + 1
print("acc 3: ", score3)

s3 = np.argmax(score3)
print('s3 : ' , s3)
s3_prob = score3[0, s3]
print('prob3', s3_prob)
print("my evaluation 3 : ", eva3)

##############################################################################################

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
commands = ['اتصل', 'ارسل', 'ابعت', 'الغاء', 'افتح', 'شغل', 'اضبط', 'قبل', 'بعد', 'ايقاف', 'اغلق', 'رسالة', 'ايميل']
answers = ['نعم', 'ايوة', 'اه', 'لا', 'تعديل']
arr_words = []



if s1_prob > s2_prob and s1_prob > s3_prob:
    for i in range(len(eva1[0])):
        if eva1[0, i] == 1:
            s = i
            arr_words.append(answers[s])
        else:
            s = ""
            arr_words.append(s)

elif s2_prob > s1_prob and s2_prob > s3_prob:
    for i in range(len(eva2[0])):
        if eva2[0, i] == 1:
            s = i
            arr_words.append(commands[s])
        else:
            s = ""
            arr_words.append(s)


elif s3_prob > s2_prob and s3_prob > s1_prob:
    for i in range(len(eva3[0])):
        if eva3[0, i] == 1:
            s = i
            arr_words.append(numbers[s])
        else:
            s = ""
            arr_words.append(s)



#word = arr_words[0]

print("word is : ", arr_words)



####################################################################################################

'''
#orignal
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(52, 22, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(5, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),   #adam
              metrics=['accuracy'])
              '''

'''model.fit(X_train, y_train_hot, batch_size=16, epochs=600, verbose=1, validation_data=(X_test, y_test_hot))

score = model.evaluate(X_Test, Y_Test_hot, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])'''
##############################################################################################


# RNN

'''
# length = 20000
drop_out_prob = 0.5
def build_tflearn_ann(length):
    input_layer = input_data(shape=[None, length, 1])


    pool_layer_1 = max_pool_1d(input_layer, 10, name='pool_layer_1')
    pool_layer_2 = max_pool_1d(pool_layer_1, 5, name='pool_layer_2')
    pool_layer_3 = max_pool_1d(pool_layer_2, 5, name='pool_layer_3')
    pool_layer_4 = max_pool_1d(pool_layer_3, 5, name='pool_layer_3')

    fully_connect_1 = fully_connected(pool_layer_3, 512, activation='relu', name='fully_connect_1',
                                      weights_init='xavier', regularizer="L2")

    fully_connect_2 = fully_connected(pool_layer_2, 512, activation='relu', name='fully_connect_2',
                                      weights_init='xavier', regularizer="L2")

    fully_connect_3 = fully_connected(pool_layer_1, 512, activation='relu', name='fully_connect_3',
                                      weights_init='xavier', regularizer="L2")

    fully_connect_4 = fully_connected(pool_layer_4, 512, activation='relu', name='fully_connect_3',
                                      weights_init='xavier', regularizer="L2")
    # Merge above layers
    merge_layer = tflearn.merge_outputs([fully_connect_1, fully_connect_2, fully_connect_3, fully_connect_4])
    # merge_layer = tflearn.merge_outputs(
    #     [fully_connect_1, fully_connect_2, fully_connect_3, fully_connect_4, fully_connect_5])
    # merge_layer = tflearn.merge_outputs(
    #     [fully_connect_1, fully_connect_2, fully_connect_3, fully_connect_4, fully_connect_5, fully_connect_6,
    #      fully_connect_7, fully_connect_8, fully_connect_9, fully_connect_10])
    drop_2 = dropout(merge_layer, 0.25)


    fc_layer_4 = fully_connected(drop_2, 2048, activation='relu', name='fc_layer_4', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_2 = dropout(fc_layer_4, drop_out_prob)


    fc_layer_5 = fully_connected(drop_2, 1024, activation='relu', name='fc_layer_5', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_3 = dropout(fc_layer_5, drop_out_prob)

    fc_layer_6 = fully_connected(drop_3, 128, activation='relu', name='fc_layer_5', regularizer='L2',
                                 weights_init='xavier', weight_decay=0.001)
    drop_4 = dropout(fc_layer_6, drop_out_prob)

    # Output
    fc_layer_2 = fully_connected(drop_4, 5, activation='softmax', name='output')
    network = regression(fc_layer_2, optimizer='adam', loss='softmax_categorical_crossentropy', learning_rate=0.0001,
                         metric='Accuracy')
    model = tflearn.DNN(network)
    return model


model = build_tflearn_ann(X_train.shape[1])

history = model.fit(X_train, y_train_hot, n_epoch=500,
          shuffle=True,
          validation_set=(X_test, y_test_hot),
          show_metric=True,
          batch_size=64)
#model.save(ann_model_dir+'Bee_audio_ANN.tfl')
'''
#######################################################################################################################

# urban model
'''
model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(52,22,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(13,activation="softmax"))

'''
