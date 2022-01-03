import numpy as np
import argparse as ap
import gui
import os
import logging

import real_time_detection
from train_model import Train_Model
from train_model import Train_Model_Infrasound

import warnings
warnings.filterwarnings('ignore')

''' Future Notes '''
#TODO: Verify Path and filename in argparse

# logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename=os.path.join(os.curdir, 'logs', "results.log"), level='INFO')
# logger = logging.getLogger(__name__)

class drone_detection:
    def __init__(self):
        # ML models for infrasound and non infrasound input data.
        self.nonInfModel = None
        self.InfModel = None
        pass

    # Run the model in real time detection mode with the non infrasound model, 
    # which will take audio from the microhphone and show the model results
    # in real time.
    def real_time_norm(self,model_name):
        print('Running Real Time Non-Infrasound Drone Detection')
        real_time_detection.run_real_time(model_name, False)
       # real_time_norm.run_real_time_nonInfr()

    # Run the model in real time detection mode with the infrasound model.
    def real_time_infr(self,model_name):
        print('Running Real Time Non-Infrasound Drone Detection')
        real_time_detection.run_real_time(model_name, True)
        pass
    # Retrain the model with new data by giving a path to the audio data from
    # the command line.
    def retrain_infr(self, path, model_name):
        model = Train_Model_Infrasound(path, model_name)
        valid_flag = model.ML_handler()
        if(valid_flag):
            logging.error(f"{model_name} failed to train")
        else:
            logging.info(f"{model_name} trained successfully")
        return 

    def retrain_norm(self, path, model_name):
        model = Train_Model(path, model_name)
        valid_flag = model.ML_handler()
        if(valid_flag):
            logging.error(f"{model_name} failed to train")
        else:
            logging.info(f"{model_name} trained successfully")
        return 

    def launch_ui(self):
        gui.launch_gui()

if __name__ == '__main__':

    # Command line argument configuration
    parser = ap.ArgumentParser(description='Drone Detect')
    parser.add_argument('-d', '--detect', type=str, help='Launch real time detection.',metavar='Model_Name')#,default='RealTime_Default_Norm')
    parser.add_argument('-i', '--infrasound', help='Specify if infrasound.')
    parser.add_argument('-r', '--retrain', type=str, help='Path to new data set.', metavar='PATH')
    parser.add_argument('-n', '--name', type=str, help='Add name for new trained model.', metavar='NAME')
    #parser.add_argument('-g', '--GUI',action='store_true', help='Launch Program with Graphical User Interface')

    args = parser.parse_args()
    drone_detect = drone_detection()

    # Real time detection models will run by default, with non-infrasound being the default model.
    if(args.retrain):
        if(args.infrasound):
            drone_detect.retrain_infr(args.retrain, args.name)
        drone_detect.retrain_norm(args.retrain, args.name)
    elif (args.infrasound and args.detect):
        # if(args.detect)
        drone_detect.real_time_infr(args.detect)
    elif (args.detect):
        drone_detect.real_time_norm(args.detect)

    else: 
        drone_detect.launch_ui()
