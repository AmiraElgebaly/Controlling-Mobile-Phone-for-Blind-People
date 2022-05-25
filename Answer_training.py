from keras.utils import to_categorical
import Model_CNN_LSTM as md
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import Preprocessing as pr

DATA_PATH1 = "data/train-answers/"

#########################################################################
#pr. save_data_to_array(path=DATA_PATH1, ext='.npy1')


X_train1, X_test1, y_train1, y_test1 = pr.get_train_test(path=DATA_PATH1, ext='.npy1.npy')

X_train1 = X_train1.reshape(X_train1.shape[0], 52, 22)
X_test1 = X_test1.reshape(X_test1.shape[0], 52, 22)

print("X_train", "y_train_hot")
y_train_hot1 = to_categorical(y_train1)
print(X_train1.shape, y_train_hot1.shape)

print("X_test", "y_test_hot")
y_test_hot1 = to_categorical(y_test1)
print(X_test1.shape, y_test_hot1.shape)

##############################################################################################


input_dim = (52, 22)
class1 = 5

model = md.cnn_lstm(input_dim, class1)

sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train1, y_train_hot1,
          batch_size=16, epochs=100,
          validation_data=(X_test1, y_test_hot1)
          )
model.save('my_model1.h5')
print("model1 saved")

##############################################################################################
