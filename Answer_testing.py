from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import Preprocessing as pr
import numpy as np
import asr

Answers = ['نعم', 'ايوة', 'اه', 'لا', 'تعديل']

file_name1 = 'test-answers'
file_name4 = 'yes'

print('test answers')
print('test asr')

X_Test1, Y_Test_hot1 = pr.test_data(file_name1)
X_Test4, Y_Test_hot4 = pr.test_data(file_name4)
print('Xtest4', X_Test4.shape)

sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

model = load_model('my_model1.h5')
print('model loaded successfully |_|_||_|')

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


#region Evaluate testing records

score1 = model.predict(X_Test4)
test_y_predictions1 = model.predict(X_Test4)

print(" Y prediction : ", test_y_predictions1.shape)
eva1 = np.zeros((2, Y_Test_hot1.shape[1]))

for index in range(test_y_predictions1.shape[0]):
    for c in range(Y_Test_hot1.shape[1]):
        if np.argmax(Y_Test_hot1[index]) == c:
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
word1 = Answers[s1]
print('The word is : ', word1)
print("my evaluation 1 : ", eva1)
# print("history :  ",history)


#endregion

#region Evaluate captured voice containning any class from answers

Folder_path = 'record.wav'
asr_file = "asr"
asr.get_command() # capture voice and save it in record.wav
asr.segmentation(Folder_path, minSilenceTime=0.35) # out the segmentations into "asr" Folder
X_Test_asr, Y_Test_asr = pr.test_data(asr_file)
# print(len(X_Test_asr))
# print(len(Y_Test_asr))

for i, _ in zip(X_Test_asr, Y_Test_asr):
    i_reshaped = i.reshape(1, 52, 22)
    print('ixtest', i.shape)
    score = model.predict(i_reshaped) # same size as number of classes with submission = 1
    s = np.argmax(score) # get index of max value 
    s_prob = score[0, s] # get the probaility
    print('s : ', s)
    print('prob', s_prob)
    word = Answers[s] # get the word
    print('The word is : ', word)

#endregion
