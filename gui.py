'''
    This module initates a Graphical user iterface that a user can interact with to control all functionality of the project.

    To use this module in other modules:
       1) import gui.py
       1) launch_gui()
'''

import tkinter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import real_time_detection
from train_model import Train_Model
from train_model import Train_Model_Infrasound
import threading

def launch_gui():
    '''
        Functions to call real-time detection and retraining functions for both infrasound and non-infrasound

    '''

    def rt_infrasound(model_name = None):
        real_time_detection.run_real_time(model_name,infrasound = True)
        print('Real Time Infrasound')

    def rt_normal(model_name = None):
        # t = threading.Thread(target=real_time_detection.run_real_time, args=(model_name,False))
        # t.start()
        real_time_detection.run_real_time(model_name,infrasound = False)
        print('Real Time Normal')
        print(model_name)

    def train_infrasound(path, name):
        model = Train_Model_Infrasound(path, name)
        model_val = model.ML_handler()
        print('Train Infrasound')
        print('Path: ', path, 'Name:', name)

        train_done_inf = tkinter.Toplevel(window)
        train_done_inf .geometry("200x100")

        t1 = tkinter.Text(train_done_inf, wrap = None)
        t1.insert(tkinter.END,'\nMODEL TRAINING COMPLETE!') 
        t1.insert(tkinter.END,'\nSound Type: Infrasound') 
        t1.pack(side='right', fill='x')
        train_done_inf.mainloop()

    def train_normal(path, name):
        model = Train_Model(path, name)
        model_val = model.ML_handler()
        print('Train Normal')
        print('Path: ', path, 'Name:', name)

        train_done_norm = tkinter.Toplevel(window)
        train_done_norm .geometry("200x100")

        t1 = tkinter.Text(train_done_norm , wrap = None)
        t1.insert(tkinter.END,'\nMODEL TRAINING COMPLETE!') 
        t1.insert(tkinter.END,'\nSound Type: Non-Infrasound') 
        t1.pack(side='right', fill='x')
        train_done_norm.mainloop()
    
    ###################################################################################################
    '''
        Create Window for GUI
    '''
    window = tkinter.Tk(className=' Drone Detect')
    window.geometry("1000x600")
    window.configure(background="#3C3744")

    ###################################################################################################
    '''
        Functions to enable selecting in list boxes
    '''
    def CurSeletLog(event):
        widget = event.widget
        selection=widget.curselection()
        picked = widget.get(selection[0])
        # print(picked)
        
        filename = str(picked)
        display_log = tkinter.Toplevel(window)
        display_log.geometry("600x400")

        # Create Scrollbar
        v = tkinter.Scrollbar(display_log)
        v.pack(side = 'right', fill = 'y')

        curr_path = os.getcwd()
        log_path_sel = curr_path+'/DetectionLogs/'+filename
        data_file = open(log_path_sel)
        data = data_file.read()
        data_file.close()

        log_text = tkinter.Text(display_log, wrap = None,yscrollcommand = v.set)
        log_text.insert(tkinter.END,data) 
        log_text.pack(side='right', fill='x')
        v.config(command=log_text.yview)
        # curr_path = os.getcwd()
        # log_path_sel = curr_path+'/DetectionLogs/'+filename
        # data_file = open(log_path_sel)
        # data = data_file.read()
        # data_file.close()

        # log_text = tkinter.Label(display_log, text = data,anchor="e")
        # log_text.pack()

    def CurSeletModel(event):
        widget = event.widget
        selection=widget.curselection()
        model = widget.get(selection[0])
        if 'normal' in model:
            print('NORMAL')
            rt_normal(model)
        elif 'infrasound' in model:
            print('INFRASOUND')
            rt_infrasound(model)
        # rt_normal(model) 

    # ###################################################################################################
    # '''
    #     Create Window for GUI
    # '''
    # window = tkinter.Tk(className=' Drone Detect')
    # window.geometry("1200x600")
    # window.configure(background="#3C3744")

    ###################################################################################################
    '''
        LIST BOXES FOR MODEL AND LOG SELECTIONS and Functions to refresh the list logs 
    '''
    log_list = tkinter.Listbox(window,width=35, height=20)
    log_list.bind('<<ListboxSelect>>',CurSeletLog)
    log_list.place(relx = 0.8,rely = 0.5,anchor ='center')

    model_list = tkinter.Listbox(window,width=35, height=20)
    model_list.bind('<<ListboxSelect>>',CurSeletModel)
    model_list.place(relx = 0.2,rely = 0.5,anchor ='center')

    def refresh_list_logs():
        curr_path = os.getcwd()
        log_path = curr_path+'/DetectionLogs/'
        dir_list = os.listdir(log_path)
        # print(dir_list)
        log_list.delete(0, "end")
        for file in dir_list:
            if(file=='.DS_Store'):
                continue
            log_list.insert("end", file)

    def refresh_list_models():
        curr_path = os.getcwd()
        mod_path = curr_path+'/Models/'
        dir_list = os.listdir(mod_path)
        # print(dir_list)
        model_list.delete(0, "end")
        for file in dir_list:
            if(file=='.DS_Store'):
                continue
            model_list.insert("end", file)


    ###################################################################################################
    '''
        REAL TIME DETECTION LABELS
    '''
    real_time = tkinter.Label(window,bg="#FBFFF1",fg='black',text ='Real Time Detection',font=("Arial", 28)
        ,borderwidth=2,relief="solid")
    real_time.place(relx = 0.2,rely = 0.1,anchor ='center')

    rt_instruct_label = tkinter.Label(window,bg="#3C3744",fg='white',text ='Select Model To Start\n Real Time Detection',font=("Arial", 14)
        ,borderwidth=0,relief="solid")
    rt_instruct_label.place(relx = 0.2,rely = 0.16,anchor ='center')

    ###################################################################################################
    '''
        Model Retraining Buttons and Labels and entry 
    '''
    train = tkinter.Label(window,bg="#FBFFF1",fg='black',text ='Retrain Model',font=("Arial", 28)
        ,borderwidth=2,relief="solid")
    train.place(relx = 0.5,rely = 0.1,anchor ='center')

    rt_instruct_label = tkinter.Label(window,bg="#3C3744",fg='white',text ='Enter Path to Dataset\n and Select Data Type',font=("Arial", 14)
        ,borderwidth=0,relief="solid")
    rt_instruct_label.place(relx = 0.5,rely = 0.16,anchor ='center')

    path_entry = tkinter.Entry(window)
    path_entry.place(relx = 0.5,rely = 0.33,anchor ='center')

    path_label = tkinter.Label(window,bg="#3C3744",fg='white',text ='New Dataset Path',font=("Arial", 14)
        ,borderwidth=0,relief="solid")
    path_label.place(relx = 0.5,rely = 0.28,anchor ='center')

    model_name = tkinter.Entry(window)
    model_name.place(relx = 0.5,rely = 0.45,anchor ='center')

    model_name_label = tkinter.Label(window,bg="#3C3744",fg='white',text ='Model Name',font=("Arial", 14)
        ,borderwidth=0,relief="solid")
    model_name_label.place(relx = 0.5,rely = 0.40,anchor ='center')

    def get_path_and_name_norm():
        print('Called')
        path = path_entry.get()
        name = model_name.get()
        train_normal(path, name)
    
    def get_path_and_name_inf():
        print('Called')
        path = path_entry.get()
        name = model_name.get()
        train_infrasound(path, name)

    tr_inf = tkinter.Button(window, text ="Infrasound", command = lambda: get_path_and_name_inf(),font=("Arial", 20)
        ,highlightbackground='#3C3744')
    tr_inf.place(relx = 0.5,rely = 0.58,anchor ='center')

    tr_norm = tkinter.Button(window, text ="Non-Infrasound", command = lambda: get_path_and_name_norm(),font=("Arial", 20)
        ,highlightbackground='#3C3744')
    tr_norm.place(relx = 0.5,rely = 0.63,anchor ='center')

    ###################################################################################################
    '''
        EXIT BUTTON
    '''
    exit = tkinter.Button(window, text = "Exit",command = window.destroy,highlightbackground='#3C3744')
    exit.place(relx = 0.92,rely = 0.95,anchor ='center')

    ###################################################################################################
    '''
        Function to print help window
    '''
    def help_window():
        display_help = tkinter.Toplevel(window)
        display_help.geometry("700x400")

        text = "Welcome to the Drone Detect GUI!\n\n"
        help_text_top = tkinter.Label(display_help, text = text,font=("Arial", 20))
        help_text_top.pack()

        text1 = "LAUNCHING REAL TIME DETECTION:\n \
            1. To launch real time detection, first refresh the model selection list.\n \
            2. Select model to be used in real time detection, real time detection will begin. \n\
            3. Sound type will be based on model selected \n\n\n"
        text2 ="MODEL RETRAINING:\n \
            1. Enter the path to the directoy containing new dataset\n \
            2. Enter a name for the model to be saved as after training.\n \
            3. Select the type of audio (infrasound or non infrasound).\n \
            4. Model will begin to train automatically. Pop up will appear when finished.\n"
        text3 = "\nDRONE DETECTION LOGS:\n \
            1. To view the past drone detection logs, first refresh the detection log list.\n \
            2. Select the log file and the contents will be displayed."
    
        # help_text = tkinter.Label(display_help, text = text,font=("Arial", 14))
        # help_text.pack()
        help_text = tkinter.Text(display_help,wrap='word')
        # help_text.pack()
        help_text.insert(tkinter.END, text1)
        help_text.insert(tkinter.END, text2)
        help_text.insert(tkinter.END, text3)
        help_text.pack(side='top', fill='x')

    '''
        HELP BUTTON
    '''
    help = tkinter.Button(window, text = "Help",command = lambda: help_window(),highlightbackground='#3C3744')
    help.place(relx = 0.05,rely = 0.95,anchor ='center')

    ###################################################################################################
    '''
        DETECTION LOGS
    '''
    log_label = tkinter.Label(window,bg="#FBFFF1",fg='black',text ='Detection Logs',font=("Arial", 28)
        ,borderwidth=2,relief="solid")
    log_label.place(relx = 0.8,rely = 0.1,anchor ='center')

    det_instruct_label = tkinter.Label(window,bg="#3C3744",fg='white',text ='Select Log to View\nDetection History'
        ,font=("Arial", 14),borderwidth=0,relief="solid")
    det_instruct_label.place(relx = 0.8,rely = 0.16,anchor ='center')

    ###################################################################################################
    '''
        REFRESH BUTTONS
    '''
    refresh_log = tkinter.Button(window, text = "Refresh",command = lambda: refresh_list_logs(),highlightbackground='#3C3744')
    refresh_log.place(relx = 0.8,rely = 0.82,anchor ='center')

    refresh_model= tkinter.Button(window, text = "Refresh",command = lambda: refresh_list_models(),highlightbackground='#3C3744')
    refresh_model.place(relx = 0.2,rely = 0.82,anchor ='center')


    window.mainloop()

# launch_gui()
