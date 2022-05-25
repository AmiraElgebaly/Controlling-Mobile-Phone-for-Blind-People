from keras.utils import to_categorical
import Model_CNN_LSTM as md
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import Preprocessing as pr

DATA_PATH3 = "data/train-numbers/"

#########################################################################

# pr.save_data_to_array(path=DATA_PATH3, ext='.npy3')

X_train3, X_test3, y_train3, y_test3 = pr.get_train_test(path=DATA_PATH3, ext='.npy3.npy')

X_train3 = X_train3.reshape(X_train3.shape[0], 52, 22)
X_test3 = X_test3.reshape(X_test3.shape[0], 52, 22)
# X_Test = X_Test.reshape(X_Test.shape[0], 20, 11, 1)
print("X_train", "y_train_hot")
y_train_hot3 = to_categorical(y_train3)
print(X_train3.shape, y_train_hot3.shape)

print("X_test", "y_test_hot")
y_test_hot3 = to_categorical(y_test3)
print(X_test3.shape, y_test_hot3.shape)

##############################################################################################


input_dim = (52, 22)
class3 = 10

model = md.cnn_lstm(input_dim, class3)

sgd = SGD(lr=0.00001, clipnorm=1.0)
adam = Adam(lr=1e-4, clipnorm=1.0)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train3, y_train_hot3,
          batch_size=16, epochs=62,
          validation_data=(X_test3, y_test_hot3)
          )
model.save('my_model3.h5')
print("model1 saved")

##############################################################################################
