import librosa
import matplotlib.pyplot as plt
from librosa import display
import noisereduce as nr
import numpy as np
from tqdm import tqdm
import os

import speech_recognition as sr
"""
Capture the voice and save it in record.wav
"""
def get_command():
    r = sr.Recognizer()
    r.pause_threshold = 1.1
    with sr.Microphone() as source:
        print("say any thing ........................")
        audio = r.listen(source=source, phrase_time_limit=7)
        try:
            text = r.recognize_google(audio, None, language="ar-EG")
            print("you say : {}".format(text))
            with open('record.wav', 'wb') as f:
                f.write(audio.get_wav_data())

        except:
            print("can't recognize.")

        text_file = open("record.txt", "w",encoding="utf-8")
        text_file.write(text)
        text_file.close()


Folder_path = 'record.wav'

""""
work on record saved in record.wav and pre_process and segment it and save the segments into "asr" Folder
"""
def segmentation(Folder_path, plot_org_signal=False, plot_after_denoise=False, plot_after_DownSampling=False,
                 desired_SR=16000,
                 print_org_signal=False, print_after_denoise=False, print_after_DownSampling=False,
                 minSilenceTime=0.40):
    correctRecords = 0
    # for recordName in tqdm(os.listdir(Folder_path)):
    name = Folder_path.split('.')
    audioName, audioFormat = name[0], name[1]
    print("audio name : ", audioName, "--->audio format", audioFormat)

    # Read file
    # path = os.path.join(Folder_path, )  # lazm 2geeb al path bta3 al sora 3n taree2 2dmg (train/imgname)
    data, sample_rate = librosa.load(Folder_path)

    # region print original signal
    if (print_org_signal == True):
        print("original signal : ")
        print("min : ", min(data))
        print("max : ", max(data))
        print("silence : ", min(abs(data)))
        print("samplr_rate : ", sample_rate)

    # endregion

    # region plot original signal
    if (plot_org_signal == True):
        plt.figure()
        librosa.display.waveplot(y=data, sr=sample_rate)
        plt.xlabel("(original signal) Time (seconds) -->")
        plt.ylabel("Amplitude")
        plt.show()
    # endregion

    print("___________________________________________________________________")

    # region noise removal
    # select section of data that is noise

    noisy_part = data
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True, )
    librosa.output.write_wav('./manal.wav', reduced_noise, sample_rate)

    # region print after denoise
    if (print_after_denoise == True):
        print("       after noise removal     ")
        print("reduced noise : ", reduced_noise.shape)
        print("min : ", min(reduced_noise))
        print("max : ", max(reduced_noise))
        print("silence : ", min(abs(reduced_noise)))
    # endregion

    # region plot after noise removal
    if (plot_after_denoise == True):
        plt.figure()
        librosa.display.waveplot(y=reduced_noise, sr=sample_rate)
        plt.xlabel("(after noise removal) Time (seconds) -->")
        plt.ylabel("Amplitude")
        plt.show()
    # endregion

    # endregion

    # region normalize
    # reduced_noise=librosa.util.normalize(reduced_noise, axis=0)

    # region plot after normalization
    # plt.figure()
    # librosa.display.waveplot(y=reduced_noise,sr=sample_rate)
    # plt.xlabel("Time (seconds) -->")
    # plt.ylabel("Amplitude")
    # plt.show()
    # endregion

    # endregion

    print("___________________________________________________________________")

    # region down samping

    sampledData, our_sr = librosa.core.resample(reduced_noise, orig_sr=sample_rate, target_sr=desired_SR,
                                                res_type='kaiser_best'), desired_SR

    # region print after down sampling
    if (print_after_DownSampling == True):
        print("       after down sampling     ")
        print("sampledData : ", sampledData.shape)
        print("min : ", min(sampledData))
        print("max : ", max(sampledData))
        print("silence : ", min(abs(sampledData)))
    # endregion

    # region plot after down sampling
    if (plot_after_DownSampling == True):
        plt.figure()
        librosa.display.waveplot(y=sampledData[0:80000], sr=our_sr)
        plt.xlabel("(after down sampling) Time (seconds) -->")
        plt.ylabel("Amplitude")
        plt.show()
    # endregion

    # endregion

    segments = np.zeros((100, 2))
    counter = 0
    start = 0
    end = 0
    words = 0

    num_of_silence = minSilenceTime * our_sr
    silence = min(abs(sampledData))
    # silenceFactor=9000000
    # max_thresh=(0.0009*maxi_sampledData)/0.04982323
    # max_thresh=max(sampledData[0:int(0.6*our_sr)]) #7lw m3 male13 , 9
    # max_thresh=max(sampledData[int((sampledData.shape[0]-1)-0.5*our_sr):int(sampledData.shape[0]-1)])
    max_thresh = 0.005
    min_thresh = -(max_thresh)

    print("min threshold : ", min_thresh, "  --> max threshold : ", max_thresh)

    # segment the record and save the segmentation into asr folder
    for index in range(sampledData.shape[0]):

        if (sampledData[index] >= min_thresh and sampledData[index] <= max_thresh):
            # print("ana silence"," --> time : ",index/our_sr," --> index : ",index,"  --> sampled data : ",sampledData[index])
            if (words == 0 and start == 0):
                continue
            elif (counter == 0):
                end = index
                counter += 1
            else:
                counter += 1

        else:
            # print("ana no silence", " --> time : ", index / 16000)
            if (counter == 0 and start == 0):
                start = index
            else:
                counter = 0
        if (counter >= num_of_silence and start != 0):
            # h2t3 mn start l end

            segments[words, 0] = start
            segments[words, 1] = end + int(num_of_silence)

            words += 1
            word = sampledData[start:end + int(num_of_silence)]
            path = "asr/" + str(words-1)
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s" % path)
            librosa.output.write_wav(('asr/'+str(words-1)+"/" + '{}.' + audioFormat).format(words - 1), word, our_sr)

            start = 0
            counter = 0
        if (index == sampledData.shape[0] - 1 and start != 0):
            print("hna")
            segments[words, 0] = start
            segments[words, 1] = end + int(num_of_silence)
            words += 1

            word = sampledData[start:end + int(num_of_silence)]

            path = "asr/" + str(words-1)
            try:
                os.makedirs(path)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s" % path)

            librosa.output.write_wav(('asr/'+ str(words-1) + "/" + '{}.' + audioFormat).format(words - 1), word,
                                     our_sr)

            start = 0
            counter = 0

        if (words >= 28):
            correctRecords = correctRecords + 1

        print("words : ", words)
    print("corrected num : ", correctRecords)

    # plot
    # plt.figure()
    # librosa.display.waveplot(y=sampledData[int(segments[8,0]):int(segments[8,1])],sr=16000)
    # plt.xlabel("(original signal) Time (seconds) -->")
    # plt.ylabel("Amplitude")
    # plt.show()
    #
    # plt.figure()
    # librosa.display.waveplot(y=sampledData[int(segments[8, 0]):int(segments[8, 1]+16000)], sr=16000)
    # plt.xlabel("(original signal) Time (seconds) -->")
    # plt.ylabel("Amplitude")
    # plt.show()


"""------***********************************************************************************************---------------"""
# get_command()
# segmentation(Folder_path, minSilenceTime=0.35)
