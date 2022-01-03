'''
    Functuon to Log Drone detections for Duration of time program runs.
'''

from datetime import datetime 
from datetime import date
import os
import os.path


class detect_logging:

    def __init__(self, sound_type):
        curr_path = os.getcwd()
        log_path = curr_path+'/DetectionLogs/'

        count = 0
        count_dir = os.listdir(log_path)
        for file in count_dir:
            count +=1
        num_files = len([name for name in os.listdir(log_path) if os.path.isfile(name)])
        # Get Date and Time
        now = datetime.now()
        curr_time = now.strftime("%H-%M-%S")
        today = date.today()
        curr_date = today.strftime("%m.%d.%y")
        self.type = sound_type
        self.filename = log_path+str(count)+' | '+curr_date +' | '+sound_type+'.txt'
        #print(self.filename)
        self.f = open(self.filename,"w")
        start_msg = str('Log of Drone Detections\nSound Type: '+self.type+'\nDate: '+curr_date+'\nTime: '+curr_time+'\n\n')
        self.f.write(start_msg)
        self.f.close()


    # Function to log events 
    def log_event(self):
        self.f = open(self.filename,"a")
        now = datetime.now()
        curr_time = now.strftime("%H:%M:%S")
        today = date.today()
        curr_date = today.strftime("%m.%d.%y")
        detection = curr_date +'|'+ curr_time+'|'+self.type+'\t Drone Detected!\n'
      #  print("detection_logged")
        self.f.write(detection)
        self.f.close()

    def end_logging(self):
      #  print("file closed")
        self.f.close()


# # Code To get current Time
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)

# # Code to get current date 
# today = date.today()
# # mm/dd/y
# d3 = today.strftime("%m/%d/%y")

# logger_inst = detect_logging('Normal')
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.log_event()
# logger_inst.end_logging()

