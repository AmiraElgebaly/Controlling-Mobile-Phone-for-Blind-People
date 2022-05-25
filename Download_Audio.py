import socketio
from tqdm import tqdm
import requests
import pickle
from flask import Flask,Response , request , flash , url_for,jsonify
from flask import Flask
import numpy as np
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import asr
import Preprocessing as pr
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

Commands = ['اتصل', 'ارسل', 'ابعت', 'الغاء', 'افتح', 'شغل', 'اضبط', 'قبل', 'بعد', 'ايقاف', 'اغلق', 'رسالة', 'ايميل']
Answers = ['نعم', 'ايوة', 'اه', 'لا', 'تعديل']
Contacts = ['ايه ربيع', 'احمد رجب']

Folder_path = 'gigarecord.wav'
asr_file = "asr"

def pridect_command(model, X, Y):
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
        # print('ixtest', i.shape)
        score = model.predict(i_reshaped)
        s = np.argmax(score)
        word_index.append(s)
        s_prob = score[0, s]
        print('s : ', s)
        print('prob', s_prob)
        # word = Answers[s]
        # print('The word is : ', word)

    return word_index

app = Flask(__name__)
socketio = SocketIO(app)
import numpy as np
@app.route('/')
@app.route('/urls/takeUrl',methods=['GET'])
def classify():
 app.logger.debug('Running classifier')
 Url = request.args.get('url')
 response = requests.get(Url, stream=True)

 with open("gigarecord", "wb") as handle:
     for data in tqdm(response.iter_content()):
         handle.write(data)
 print('are you ready !!!')
 # asr.get_command()
 asr.segmentation(Folder_path, minSilenceTime=0.45)  # put on asr Folder
 X_Test_asr, Y_Test_asr = pr.test_data(asr_file)    # function to read x,y from asr folder
 classes = pridect_command('my_model2.h5', X_Test_asr, Y_Test_asr)
 # print("indecies", classes)
 word = classes[0]
 # print('The command is : ', Commands[word])
 #
 if Commands[word] == 'اتصل':
     file = open('record.txt', 'r', encoding='utf-8')
     line = file.read()
     words = line.split()
     contact = words[-2:]
     contact_name = ''
     for i in contact:
         contact_name += i
         contact_name += ' '

 return "Successfully"+"+"+Commands[word]+"+"+contact_name+"+"+line

#http://127.0.0.1:5000/urls/takeUrl?url=https://firebasestorage.googleapis.com/v0/b/blindpeoplegp.appspot.com/o/record.wav?alt=media&token=690d671c-125f-49e7-b28b-b2891b3262a4



###############################################################################
if __name__ == "__main__":
    socketio.run(app)
