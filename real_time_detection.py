'''
    real_time_detection.py handles the execution of real time detection for both infraosund and non-infrasound

    to begin execution and call function:

        1) import real_time_detection.py
        2) real_time_detection.run_real_time(model_name='RealTime_Default_Norm',infrasound = False):

        Can be calleed with no arguments in which case the function will run using the default non-infrasound model

        Real time deteciton can be run on any model of users choosing by adding argument for model_name, bool argument infrasound
        should be handled when function is being called (from GUI or commmand line arguments)
'''

# from IPython.display import Audio
# from IPython import display
import torch
import pyaudio
import numpy as np
import pylab
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import faulthandler; faulthandler.enable()
import sys
import select
import tkinter
tf.autograph.set_verbosity(3)
import warnings
import drone_logging

import signal 
warnings.filterwarnings('ignore')
os.system('cls||clear')


RATE = 16000
CHUNK = int(RATE) # RATE / number of updates per second

def get_prediction(data_chunk, model, infrasound):
    ''' get_prediction() returns prediction for one second clip of data 
    from stream '''
    
    labels = ['drone', 'no drone']

    # data_chunk = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
    waveform = np.float32(data_chunk)
    max_int16 = 2**15
    waveform_normalised = waveform / max_int16
    
    # Convert waveform from numpy array to tensor
    waveform_tensor = torch.from_numpy(waveform_normalised)
    waveform_tensor = tf.reshape(waveform_tensor, [1, -1])

    
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform_tensor), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the same length
    data = tf.cast(waveform_tensor, tf.float32)
    
    # Compute spectrogram
    spectrogram = tf.signal.stft(data, frame_length=255, frame_step=128)
    
    spectrogram_abs = tf.abs(spectrogram)

    prediction = model.predict(spectrogram_abs)
    y_pred = np.argmax(prediction)

    return prediction

#model_name='normal_training_demo'
#model_name='normal_RealTime_Default'
def run_real_time(model_name='normal_RealTime_Default',infrasound = False):
    logger_inst = drone_logging.detect_logging('Normal')

    curr_path = os.getcwd()
    path = '/Models/'

    model_path = curr_path + path + model_name


    print('Running Real Time Detection for model: ', model_name)

    model = tf.keras.models.load_model(model_path)
    labels = ['drone', 'no drone']

    print('\nStarting Real-Time Drone Detection\n\n')

    fig, ax = plt.subplots()
    
    p=pyaudio.PyAudio()

    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,frames_per_buffer=CHUNK) #, input_device_index=0)
    stream.start_stream()

    run = True
    while run == True:
        try:

            data_chunk = np.frombuffer(stream.read(CHUNK),dtype=np.int16)

            # Get Prediction based on model type
            if(infrasound == True):
                # prediction = get_prediction(stream, model, True)
                prediction = get_prediction(data_chunk, model, False)
            elif(infrasound == False):
                # prediction = get_prediction(stream, model, False)
                prediction = get_prediction(data_chunk, model, False)
            #print(prediction)

            if(prediction[0][0] > prediction[0][1]-1):
                logger_inst.log_event()
                print('DRONE DETECTED')
                # log_text.insert(tkinter.END,"Here\n") 
                # log_text.update()
            elif(prediction[0][0] > prediction[0][1]-2):
                logger_inst.log_event()
                print('Drone Presence Likely')
            elif(prediction[0][0] < prediction[0][1]):
                continue
                #print('no drone')
            # signal.signal(signal.SIGINT, exit_handler)
        #print(prediction)
        except KeyboardInterrupt:  
            logger_inst.end_logging()
            run = False
            #logger_inst.end_logging()

    # except KeyboardInterrupt:
    #     print("\nEnding Real-Time Detection\n\n")
    #     logger_inst.end_logging()
    #     sys.exit()

    # rt_window.mainloop()
    #print("still here?")

    stream.stop_stream()
    stream.close()
    p.terminate()
    logger_inst.end_logging()
    return 0

#run_real_time()
