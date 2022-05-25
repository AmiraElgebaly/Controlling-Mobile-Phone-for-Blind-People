from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import Preprocessing as pr
import numpy as np
import asr

file_name2 = 'test-Commands'
file_name4 = 'yes'
Commands = ['اتصل', 'ارسل', 'ابعت', 'الغاء', 'افتح', 'شغل', 'اضبط', 'قبل', 'بعد', 'ايقاف', 'اغلق', 'رسالة', 'ايميل']

print('test Commands')
print('test asr')


X_Test2, Y_Test_hot2 = pr.test_data(file_name2)

X_Test4, Y_Test_hot4 = pr.test_data(file_name4)



sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

model = load_model('my_model2.h5')
print('model loaded successfully')

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

#region Evaluate testing records

score2 = model.predict(X_Test4)
test_y_predictions2 = model.predict(X_Test4)

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
print('s2 : ', s2)
print('prob2', s2_prob)
print("my evaluation 2 : ", eva2)
word2 = Commands[s2]
print('The word is : ', word2)

#endregion

#region Evaluate captured voice containning any class from commands
Folder_path = 'record.wav'
asr_file = "asr"
asr.get_command()
asr.segmentation(Folder_path, minSilenceTime=0.45)
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
    word = Commands[s]
    print('The word is : ', word)

#endregion