import Model_CNN_LSTM as md
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import Preprocessing as pr
import numpy as np

import asr

file_name3 = 'test-numbers'
file_name4 = 'yes'
Numbers = ['صفر', 'واحد', 'اثنين', 'ثلاثة', 'اربعة', 'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة']

print('test Numbers')
print('test asr')


X_Test3, Y_Test_hot3 = pr.test_data(file_name3)

X_Test4, Y_Test_hot4 = pr.test_data(file_name4)



sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

model = load_model('my_model3.h5')
print('model loaded successfully')

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


#region Evaluate testing records

score3 = model.predict(X_Test4)
test_y_predictions3 = model.predict(X_Test4)
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
print('s3 : ', s3)
s3_prob = score3[0, s3]
print('prob3', s3_prob)
word3 = Numbers[s3]
print("The word is : ", word3)
print("my evaluation 3 : ", eva3)

#endregion


#region Evaluate captured voice containning any class from numbers
Folder_path = 'record.wav'
asr_file = "asr"
asr.get_command()
asr.segmentation(Folder_path, minSilenceTime=0.35)
X_Test_asr, Y_Test_asr = pr.test_data(asr_file)
# print(len(X_Test_asr))
# print(len(Y_Test_asr))
for i, j in zip(X_Test_asr, Y_Test_asr):
    i_reshaped = i.reshape(1, 52, 22)
    print('ixtest', i.shape)
    score = model.predict(i_reshaped)
    s = np.argmax(score)
    s_prob = score[0, s]
    print('s : ', s)
    print('prob', s_prob)
    word = Numbers[s]
    print('The word is : ', word)

#endregion