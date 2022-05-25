from keras.utils import to_categorical
import Model_CNN_LSTM as md
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import Preprocessing as pr

DATA_PATH2 = "data/train-Commands/"

#########################################################################


# pr.save_data_to_array(path=DATA_PATH2, ext='.npy2')


X_train2, X_test2, y_train2, y_test2 = pr.get_train_test(path=DATA_PATH2, ext='.npy2.npy')

X_train2 = X_train2.reshape(X_train2.shape[0], 52, 22)
X_test2 = X_test2.reshape(X_test2.shape[0], 52, 22)
# X_Test = X_Test.reshape(X_Test.shape[0], 20, 11, 1)
print("X_train", "y_train_hot")
y_train_hot2 = to_categorical(y_train2)
print(X_train2.shape, y_train_hot2.shape)

print("X_test", "y_test_hot")
y_test_hot2 = to_categorical(y_test2)
print(X_test2.shape, y_test_hot2.shape)

##############################################################################################


input_dim = (52, 22)
class2 = 13

model = md.cnn_lstm(input_dim, class2)

sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train2, y_train_hot2,
          batch_size=16, epochs=62,
          validation_data=(X_test2, y_test_hot2)
          )
model.save('my_model2.h5')
print("model1 saved")

##############################################################################################
