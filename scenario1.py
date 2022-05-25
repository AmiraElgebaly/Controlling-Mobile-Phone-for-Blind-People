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
import female_preprocessing as fb
import matplotlib.pyplot as plt
import asr
import Preprocessing as pr

Commands = ['اتصل', 'ارسل', 'ابعت', 'الغاء', 'افتح', 'شغل', 'اضبط', 'قبل', 'بعد', 'ايقاف', 'اغلق', 'رسالة', 'ايميل']
Answers = ['نعم', 'ايوة', 'اه', 'لا', 'تعديل']
Contacts = ['ايه ربيع', 'احمد رجب']

Folder_path = 'record.wav'
asr_file = "asr"

def pridect_command(model,X, Y):
    word_index = []
    sgd = SGD(lr=0.00001, clipnorm=1.0)
    adam = Adam(lr=1e-4, clipnorm=1.0)
    model = load_model(model)
    print('model loaded successfully |_|_||_|')
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    for i, j in zip(X, Y):
        i_reshaped = i.reshape(1, 52, 22)
        #print('ixtest', i.shape)
        score = model.predict(i_reshaped)
        s = np.argmax(score)
        word_index.append(s)
        s_prob = score[0, s]
        print('s : ', s)
        print('prob', s_prob)
        #word = Answers[s]
        #print('The word is : ', word)

    return word_index

print('are you ready !!!')
asr.get_command()
asr.segmentation(Folder_path, minSilenceTime=0.45)
X_Test_asr, Y_Test_asr = pr.test_data(asr_file)
classes = pridect_command('my_model2.h5', X_Test_asr, Y_Test_asr)
print("indecies", classes)
# word = classes[0]
# print('The command is : ', Commands[word])
#
# if Commands[word] == 'اتصل':
#     file = open('record.txt', 'r', encoding='utf-8')
#     line = file.read()
#     print(line)
#     words = line.split()
#     print(words)
#     contact = words[-2] + ' ' + words[-1]
#     print(contact)

# elif Commands[word] == 'ارسل':
#
# elif Commands[word] == 'ابعت':