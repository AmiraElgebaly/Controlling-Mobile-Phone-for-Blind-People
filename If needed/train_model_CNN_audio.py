import model_audio_CNN
import pickle as cPickle
import tensorflow as tf
import numpy as np
import os
'''
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# export CUDA_VISIBLE_DEVICE=0,1

pickle_path = 'pickle_data/'
print("loading pickle_data files for CNN")
with open(pickle_path+"train_data_22k_org.pickle_data", "rb") as input_file:
    x_train = cPickle.load(input_file)

with open(pickle_path+"train_labels_22k_org.pickle_data", "rb") as input_file:
    y_train = cPickle.load(input_file)

with open(pickle_path+"test_data_22k_org.pickle_data", "rb") as input_file:
    x_test = cPickle.load(input_file)

with open(pickle_path+"test_labels_22k_org.pickle_data", "rb") as input_file:
    y_test = cPickle.load(input_file)


x_train = np.row_stack([x_train, x_test])
y_train = np.row_stack([y_train, y_test])


with open(pickle_path + 'valid_data_22k_org.pickle_data', 'rb') as input_file:
    x_valid = cPickle.load(input_file)

# with open(pickle_path+"test_labels_22k_org.pickle_data", "rb") as input_file:
with open(pickle_path + "valid_labels_22k_org.pickle_data", "rb") as input_file:
    y_valid = cPickle.load(input_file)

'''




x_train = []
y_train = []


females=os.listdir('dataset/female_train')
males=os.listdir('dataset/male_train')
print(len(females))
print(len(males))
for i in females:
    print(i)
    y_train.append("0")
    x_train.append(i)
for i in males:
    print(i)
    y_train.append("1")
    x_train.append(i)

for i in range(len(x_train)):
    print(x_train[i]+" "+y_train[i])

x_train = np.array(x_train)
y_train = np.array(y_train)
print(y_train.shape)

##############
# Train CNN ##
##############
NUM_EPOCHS = 500
BATCH_SIZE = 1
MODEL = model_audio_CNN.build_tflearn_cnn(x_train.shape[0])#call
# with tf.device('/gpu:0'):

MODEL.fit(x_train, y_train, n_epoch=NUM_EPOCHS,
              shuffle=True,
              show_metric=True,
              batch_size=BATCH_SIZE)
MODEL.save('male_audio_CNN.tfl')





# ##################
# ## Evaluate CNN ##
# ##################
# tf.reset_default_graph()

# cnn_model_dir = '/home/vishal/PycharmProjects/bee_audio_project/model/CNN/Bee_audio_CNN_100.tfl'
# # cnn_model = model_audio_CNN.build_tflearn_cnn(x_test.shape[1])

# MODEL.load(cnn_model_dir, weights_only=True)
# validation_acc = MODEL.evaluate(x_test, y_test)
# print(validation_acc)