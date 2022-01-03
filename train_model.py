#!/usr/bin/python python3
import os
import pathlib


os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import logging
import warnings
import tensorflow_io as tfio

warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(os.curdir, 'logs', "results.log"), level='INFO')
logger = logging.getLogger(__name__)

''' Future Notes '''
#TODO: Add inheritance class for inf
#TODO: Make a class for testing the model

class InvalidPath(Exception):
    ''' Rasied when an InvalidPath is passed into our ML Training Service '''
    pass

class InvalidDataType(Exception):
    ''' Rasied when provided data is not a .wav file '''
    pass

class InvalidDataStructure(Exception):
    ''' Rasied when provided data is not a formatted correctly '''
    pass

class FailedTraining(Exception):
    ''' Rasied when data formatting or permissions prevent training the model '''
    pass

class FailedBuilding(Exception):
    ''' Rasied when data formatting or permissions prevent building the model  '''
    pass

class Train_Model():

    def __init__(self, path_input, model_name):
        self.labels = None
        self.path_input = path_input
        self.model_name = model_name

    def decode_audio(self, audio_binary):
        ''' Convert .WAV audio files into a numerical tensor '''

        # desired_samples should be the length of the file
      
        audio, _ = tf.audio.decode_wav(audio_binary, desired_samples = 16000)
        return tf.squeeze(audio, axis=-1)
        

    def get_label(self, file_path):
        ''' Given a file path, return parent directory name ( Label ) '''
      
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2]
   

    def get_waveform_and_label(self, file_path):
        ''' Given a scalar tensor dataset containing audio file paths ... return tensors 
        of binary audio data and its associated label '''
        
        # Get data label
        try:
            label = self.get_label(file_path)
            audio_binary = tf.io.read_file(file_path)
        except (ValueError, tf.errors.NotFoundError):
            logging.error(f'{file_path} is not a valid path')
            raise InvalidPath
        
        # Convert .WAV to binary audio tensor
        try:
            waveform = self.decode_audio(audio_binary)
        except tf.errors.InvalidArgumentError:
            logging.error(f'{file_path} is not a .WAV file, and can not be decoded')
            raise InvalidDataType
        
        return waveform, label
    
    def unpack_dataset(self, file_path):
        ''' Extracts a set of data given a file path. Returns training set, 
        validation set, and test set lists containing audio file paths '''

        # Ensure provided path is valid and TensorFlow can handle the data
        try:
            assert(os.path.exists(file_path))
            data_dir = os.path.abspath(file_path)
            logger.info(f'DATA_DIR = {data_dir}')

            # Find audio designation labels ( labels must be directory names )
            self.labels = np.array(tf.io.gfile.listdir((data_dir)))

            # Extract audio files into a list and shuffle it
            filenames = tf.io.gfile.glob(data_dir + '/*/*')
            filenames = tf.random.shuffle(filenames)
        except (AssertionError, PermissionError, tf.errors.OpError, tf.errors.NotFoundError):
            logging.error(f'{file_path} is not a valid path or contains an invalid data structure')
            raise InvalidPath
        
        num_samples = len(filenames)

        logging.debug(f'Number of total examples: {num_samples}')
        logging.debug(f'Example file tensor: {filenames[0]}')

        # Split files into training, validation, and tests sets
        training_pct = int(len(filenames)*0.7)
        val_test_pct = int(len(filenames)*0.15)

        train_files = filenames[:training_pct]
        val_files = filenames[training_pct: training_pct + val_test_pct]
        test_files = filenames[-val_test_pct:]

        return train_files, val_files, test_files

    def get_spectrogram(self, waveform):
        ''' Converts array of binary audio into the time-frequency domain,
        and returns the frequency in a tensor '''
       
        
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

        # Concatenate audio with padding so that all audio clips will be of the same length
        waveform = tf.cast(waveform, tf.float32) # Typecast each element
    # data_fade = tfio.experimental.audio.fade(waveform, fade_in=1000, fade_out=1000, mode="logarithmic")
        
        equal_length = tf.concat([waveform, zero_padding], 0)
        # equal_length = tf.concat([waveform, zero_padding], 0)

        # Convert binary audio into the time-frequency domain, returning the frequency and phase
        spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
        #freq_mask = tfio.experimental.audio.freq_mask(spectrogram, param=10)
        
        # We are only interested in the magnitude of audio frequency
        spectrogram = tf.abs(spectrogram)

        return spectrogram
        


    def get_spectrogram_and_label_id(self, audio, label):
        
        ''' Transforms the waveform dataset to have spectrogram images and their corresponding 
        labels as integer IDs '''
        

        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)

        label_id = tf.argmax(label == self.labels)
        return spectrogram, label_id
        
    def preprocess_dataset(self, files, AUTOTUNE):
        '''  Given a list of paths to .WAV audio files, return a tensor dataset with each entry containing a
        spectrogram and label for that audio path ''' 
        
        # Create a 'tensor object' dataset
        files_ds = tf.data.Dataset.from_tensor_slices(files)

        # Replace each tensor object in the dataset with a list containing the numerical tensor from each decoded .wav file path 
        #   and its corresponding label (unknown, yes_drone)
        try:
            waveform_ds = files_ds.map(self.get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        except (InvalidPath, InvalidDataType):
            raise InvalidDataStructure

        # Transform the waveform dataset to have spectrogram images and their corresponding labels as integer IDs.
        try:
            output_ds = waveform_ds.map(self.get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
        except (TypeError, ValueError, tf.errors.InvalidArgumentError):
            logging.error(f"Invalid Data Structure")
            raise InvalidDataStructure

        return output_ds
    
    def build_model(self, train_ds, input_shape, num_labels):
        ''' Build a convolutional neural network (CNN) with preprocessing layers:
                A Resizing layer to downsample the input to enable the model to train faster.
                A Normalization layer to normalize each pixel in the image based on its mean and 
                standard deviation. '''

        #  Create a Normalization layer to normalize each pixel in the image based on 
        #   its mean and standard deviation.
        norm_layer = preprocessing.Normalization()

        # Compute mean and variance of the data and store them as the layer's weights
        norm_layer.adapt(train_ds.map(lambda x, _: x))

        logging.debug(f"Labels: {num_labels}")

        # Initialize convolutional neural network (CNN)
        model = models.Sequential([
            layers.Input(shape=input_shape),
            preprocessing.Resizing(32, 32), 
            norm_layer,
            layers.Conv2D(32, 3, activation='relu'),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_labels),
        ])

        model.summary()

        # Configure the model
        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
            )
        except ValueError:
            logging.error("Failed to compile, invalid arguments for optimizer, loss or metrics.")
            raise FailedBuilding

        return model

    def train_model(self, train_ds, val_ds, model, epochs):
        ''' Given a number of training epochs, train the CNN '''
    
        try:
            model.fit(
                train_ds, 
                validation_data=val_ds,  
                epochs=epochs,
                callbacks=tf.keras.callbacks.EarlyStopping(patience=2),
                # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            )
        except(ValueError, RuntimeError):
            logging.error("Unable to fit model, mismatch between the provided input data and what the model expects")
            raise FailedTraining

        return model

    def ML_handler(self):
        print("IN HANDLER!")
        ''' Handles the pre-processing, building, and training of the Machine 
        Learning Model '''

        logger.info("Starting Training Service")

        # TODO: Specify how a new user would configure directory in a README
        # Divide dataset into ML segments and find associated labels
        try:
            train_files, val_files, test_files = self.unpack_dataset(self.path_input)
        except InvalidPath:
            return 1 # Fail

        # Produce a tensor dataset for each segment
        # Each entry in the dataset contains a spectrogram, and its associated label
        AUTOTUNE = tf.data.AUTOTUNE
        try:
            train_ds = self.preprocess_dataset(train_files, AUTOTUNE)
            val_ds = self.preprocess_dataset(val_files, AUTOTUNE)
            test_ds = self.preprocess_dataset(test_files, AUTOTUNE) #TODO: run an accuracy test
        except InvalidDataStructure:
            return 1

        # Find shape of each spectrogram array, and ammount of labels
        for spectrogram, _ in train_ds.take(1):
            input_shape = spectrogram.shape
        num_labels = len(self.labels)

        # Build the ML model
        try:
            model = self.build_model(train_ds, input_shape, num_labels)
        except FailedBuilding:
            return 1

        # Combine consecutive elements of the training and validation sets for model training
        batch_size = 64
        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)

        # Add dataset cache() and prefetch() operations to reduce read latency while training the model
        train_ds = train_ds.cache().prefetch(AUTOTUNE)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        
        # Train the Model
        try:
            model = self.train_model(train_ds, val_ds, model, 15)
        except FailedTraining:
            return 1

        # Save the model
        temp_model_name = 'normal_'+self.model_name
        model_path = os.path.join(os.getcwd(),"models", temp_model_name)
        try:
            model.save(model_path)
        except:
            logging.error(f"Unable to save the model at {model_path}")
            return 1
        # metrics = model.history
        # print(metrics)
        # metric = metrics['val_acc']
        # print(metrics)

        return 0

# This is where you should put the low_pass filter function
class Train_Model_Infrasound(Train_Model):

    pass

# Testing Model ( Delete in Final )
#emp1 = Train_Model("//Users/brianschwantes/Desktop/Infrasound", "test")
# emp1 = Train_Model("/Users/brianschwantes/Desktop/Capstone/E490-Capstone/DroneAudioDataset/Binary_Drone_Audio", "test")
# emp1.ML_handler()