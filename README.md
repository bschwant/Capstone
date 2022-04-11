# E490-Capstone

Install needed libraries: ```pip install -r requirements.txt```

## 2. Project Overview
### 2.1 Background
Surveillance and warfare tactics are becoming increasingly autonomous. The use of machines such as drones allow entities to survey and potentially attack from a safe and hard to detect distance. This project aims to eliminate these advantages by building a detection system that will identify objects at a greater distance by analyzing their infrasound data.

### 2.2 Problem Statement 
 _How can we rapidly identify and localize and object without being able to see it?_

Due to the relatively small size and remote guidance capabilities, small unmanned aerial vehicles (sUAS) - commonly called drones - are notoriously difficult to detect. Since early detections at beyond-line-of-sight ranges are critical for battlespace awareness, a reliable way of “seeing” drones needs to be discovered. One potential approach is to listen for dones and use their audio signature to identify them using their infrasound. This new approach is important for the following reasons:
Sound frequencies audible to humans do not travel far before they dissipate into heat.
Infrasound, of frequencies below 20 Hz, are inaudible to humans but travel a larger distance before dissipating into heat.

### 2.3 Use Cases
Infrasound drone detection can be used in many circumstances, one instance is a military base. An enemy drone could be flying above the base and no would know without drone detection capabilities. Drones flying undetected could prove disastrous to people within the base and it is crucial people are made aware of the presence quickly so safety measures can be taken. 

### 2.4 Project Description
This project involves detecting drones by listening for their sound characteristics from a long distance inorder to identify them in a defense scenario. The goal of our project is to create software to process audio data and identify the presence of drones based on their infrasound. Ultimately we decided to extend our project to work with both infrasound and audible sound. In order to complete this, we first had to create a reliable dataset of drone infrasound data. We then had to determine a machine learning technique suitable for rapid characterization of objects based on the auto characteristics produced by drones. Next we trained the machine learning models to detect the presence of drones rapidly and accurately based on their audible sound and infrasound. Finally, we had to package our program in a user-friendly way to be efficiently used by our client and added a GUI for users that are less technically inclined.

### 2.5 Project Objectives
1. Create Dataset of Drone Infrasound Data
2. Identify a Machine Learning Technique Suitable for Rapid Classification of Audio Data
3. Train Machine Learning Model for Detection
4. Implement Trained Model into Software Capable of Real-Time Detection

### 2.6 Pairwise Comparison Chart: Objectives

### 2.7 Key Features
1. Accurate Classification of Drones
2. Accurate Location Tracking of Drones
3. Simple User Interface to Display Output Data

### 2.8 Deliverables 
An accurate and properly trained machine learning algorithm
Easy to use packaging of our program capable of real-time drone detection.

### 2.9 Constraints 
Compile reliable data set of Infrasound Data
The dataset must be large enough to train and test a machine learning model. However, infrasound data is difficult to acquire and not widely available. The equipment needed to record and collect infrasound data is unobtainable.
Implementing a Reliable Machine Learning Algorithm 	
Because of the small scope that is infrasound, distinguishing drones amongst all of the other sources of infrasound make this challenging.
User-Friendly Program
Our software must be easy to operate and understand or it may not be effective.

### 2.10 Performance Requirements
R1: Machine Learning Model Accuracy:Train a model to detect drones that is satisfactory to our client.

R2: Drone Detection Distance: Create software able to detect drones at a distance that is satisfactory to our client

R3: Software Reliability: Software should be able to run continuously with minimal interruptions to ensure client safety.

### 2.11 Business Requirements

## 3. Basic Resources
### 3.1 Total Estimated Man-Hours
Each team member spent approximately 8 hours a week towards the project which includes group meetings and individual tasks. This time consisted of 2 hours per week dedicated to team meetings and 4-6 hours a week to complete individual tasks as well as group assignments.

### 3.1.1 Semester 1 Gantt Chart
### 3.1.2 Semester 2 Gantt Chart

### 3.2 Available Resources  
Our team has a budget of $200 from Luddy. As our project has consisted of only software up until this point, we have not used any of the budget yet. We also had the expertise of our Mentor from NSWC Crane Ryan Dowd, and Professor Suha Lasassmeh to assist during our project.

## 4. Team Management 

### 4.1 Team Meetings
The team meets three times a week via Zoom for the duration of our projects. Additional team meetings were also planned frequently at various times during the week to work as a group. The schedule for team meetings is as follows:
Sunday: 6:00 - 7:30 PM  
Monday: 9:25 - 10:40 AM
Wednesday: 9:25 - 10:40 AM 

### 4.2 Decision Making Policy 
All decisions for our project were made by a group majority vote. If our team cannot make effective decisions according to our agreed upon guidelines, group members may appeal to Professor Suha Lasassmeh. Professor Lasassmeh has the final call on all decisions.
### 4.3 Communication
Communication between group members and mentors for this project is entirely done through the use of Zoom, email, and text messaging.
### 4.4 Data Archive  
Our team stored all project related documentation in a shared Google Drive. All presentations and labs were created and stored on Google Slides and Google Docs respectively. 

## 5. Technical Design
### 5.1 Product Block Diagram

### 5.2 Block Ownership


Block A: Dataset Creation - Ethan Japundza
- Find/Create/Optimize dataset suitable for ML training.
- Create simulated real time audio data.

Block B: ML Training - Brian Schwantes
- Identify suitable ML algorithm for real time detection.
- Train ML model for drone audio detection.
- Evaluate and test performance of model.

Block C: Real Time Detection - Brian Schwantes
- Implement a trained ML model into function capable of handling real time audio data.

Block D: Model Retraining - Ethan Japundza
- Implement ML model training into function to retrain for new data.

Block E: User Interface - Andrew Gotts
-  Create UI for easy interaction between user and software to control all functionality.

While each block was owned by a member of the group, each member contributed and assisted in the completion of all blocks within the project.

## 5.3 Software Platform
We decided to use Python for all portions of our project thus far because of the countless packages available for machine learning and sound processing. The main Python packages used to complete our project are:
TensorFlow - Open-source software library for machine learning.
Keras - Open-source library that provides a Python Interface for the Tensorflow library when creating artificial Neural Networks.
Librosa - Python package for music and audio analysis.
SciPy - Open-source library containing functions for signal processing and visualization.

## 6. Similar Intellectual Property
### 6.1 US Patent 1
DRONE DETECTION AND CLASSIFICATION WITH COMPENSATION FOR BACKGROUND CLUSTER SOURCES - #10,032,464

Summary
This patent covers a system, method, and apparatus for detecting drones by receiving a digital sound sample and partitioning the digital sound sample into segments. This method includes applying a frequency and power spectral density transformation to each of the segments and determining if a drone is present.

Infringement
Our project does not infringe on the device portion of this patent, we believe our project also does not infringe on the detection methods either as they are not using infrasound for detection but rather audible sound.
### 6.2 US Patent 2
Apparatus for detecting infrasound -  #9,753,162

Summary
The apparatus in this patent is for detecting infrasound with a sound wave detector including a diaphragm dividing the inner space of a first space and a second space with microchannel with different resistance values with respect to transmission of a sound wave.
Infringement 
No our project does not infringe on this patent because we are not creating a device to detect infrasound.
### 6.3 US Patent 3
Detection Device, information input device, and watching system -  #10,733,875

Summary 
A detection device for detecting inaudible (infrasound) sound waves generated a user’s bodily motion to identify the user’s motion.

Infringement 
No our project does not infringe on this patent because we are not creating a device, nor are we attempting to detect human motion.


## 7. Ethical Considerations 
### 7.1 Ethical Concern 1: Conflicts of Interest
Our project consists of making a drone detection system for Crane using machine learning. This software and machine learning model in our project will be one of the first created to detect drones based on their infrasound data and could be very valuable to other parties besides Crane for various applications. Because software can also be used in military applications, it is important that only authorized users have access to it.

Affected Bock
All Blocks
Mitigation
The code for our project will not be made publicly accessible.
Software will be shared privately with our client Crane.
Any other parties interested in our software will have to receive explicit permission from our client before our software is shared.
### 7.2 Ethical Concern 2: ML Model Accuracy 
Depending on the application of our software and who and where it is being used, misunderstanding the accuracy of our machine learning model could put the user at risk. Because of limited training data, data may have been created by altering available sound data to be infrasound which may have an impact on accuracy. Our models accuracy may also be affected by other sources of infrasound such as planes, helicopters, seismic events, and even severe weather.

Affected Block
Machine Learning Model (B) | Brian
Mitigation
Clearly state the accuracy and limitations of the machine learning model in various conditions and possible causes of inaccuracy.
### 7.3 Ethical Concern 3: Inaccuracy due to Drone Variety
In order to train a machine learning model to detect a drone based on infrasound, large amounts of audio data for drones is needed. Because different brands/sizes drones have different sound characteristics, our machine learning model will be trained to detect drones that we have sound data for.
 
Affected Blocks
Machine Learning Model (B) | Brian
Mitigation
List drones that are detectable and undetectable by our ML model.

## 8. Safety Considerations
### 8.1 Safety Concern 1: ML Model Accuracy 
The accuracy of our machine learning model could also become a safety concern for our project. Depending on the application during the use of our software, a missed detection of a drone, or a false detection could put our users at risk. For example, if our software is used in a military setting for security, a missed detection of an enemy drone could be disastrous.

Mitigation
Because we are unable to create a model with 100% detection accuracy, we must be extremely clear and honest with our client in regard to the reliability and accuracy of our software in order for them to make appropriate decisions in regards to how they use it.
### 8.2 Safety Concern 2: Software Reliability
 The reliability of the software is extremely important to our client. If our software stops working as expected and the user is unaware, drones may approach an area undetected which would put the user and others at risk.

Mitigation
While the goal is always to make software that is 100% reliable and will never fail, this is not always possible in practice. Because of this, we will incorporate failsafes into our program in the event it stops operating as expected and will alert the user to any potential issues.


## 9. Block A: Dataset Creation

Figure 9.A: Block A 

The main objective of Block A was the creation of a reliable data set of drone audio and the preprocessing of data to be used for machine learning training. This block is also responsible for the creation of data to simulate an audio input for real time sound classification.
### 9.1 Block Requirements
Req #
Requirement
Description
A1
Infrasound Dataset
Compilation of dataset processed to extract infrasound
A2
Non-Infrasound Dataset
Compilation of dataset of audible drone data
A3
Real-Time Audio
Creation of simulated real-time drone audio 

### 9.2 Input/Output Summary 
This block has no input/output signals associated with it.
### 9.3 Cost Requirements 
There are no costs associated with this block as all of the data used was publicly available online.
### 9.4 Environmental Requirements 
There are no environmental requirements associated with this block.
### 9.5 Data Set
As the goal of this project is to detect drones using their infrasound signatures, it would be ideal to have a dataset consisting of actual drone infrasound to use when training our machine learning model. Our mentor had originally hoped to provide true drone infrasound data; however, the process of getting the data declassified in the midst of a pandemic in time to deliver to us for our project was unsuccessful. Because of this, it was up to us to compile a reliable infrasound dataset of drone data which in the ended proved to be one of the most challenging parts of our project. Throughout our first semester working on the project we discovered publicly available drone data exists only for natural events such as earthquakes, volcanic activity, and severe weather. Our mentor from Crane also taught us a lot about the equipment needed to record infrasound. This equipment is typically extremely large (the size of a football field); newer infrasound recording devices are the size of a dinner plate but prohibitivly expensive. Due to these challenges we encountered, our group made the decision to extend our project to also work with non-infrasound data as well to deliver a better product to our clients.

Due to the previously mentioned issues, in order to complete our project, our group decided to use a dataset containing commercial drone audio clips also containing random audio clips [3]. 
The dataset we ended up using contains indoor drone propeller noise recordings by Sara Al-Emadi and has also been artificially augmented with random noise clips. This dataset is part of  the 'Audio Based Drone Detection and Identification using Deep Learning’ research paper.[1] We used 16,000 audio clips in the training,validation, and testing of our machine learning model
### 9.6 Data Preprocessing
The dataset mentioned previously contained audio clips recorded using standard microphones  and were audible frequencies and not infrasound. As a result, in order to create a dataset to train our infrasound machine learning model, we had to use sound processing techniques to extract the infrasound from these clips. Peak frequencies within the true drone audio from the dataset were roughly 8000 Hz. For our project we need to focus on the infrasound which are the frequencies between 0-20 Hz. To get the infrasound data from these audio clips accomplish this, we used Python to apply a low-pass filter to all of the audio clips and save them as new files to form a pseudo-infrasound database. It is important to note, this method is not ideal due to the fact the microphones used to record this data are not designed to record frequencies this low. As a result, after sound processing, we noticed lots of noise in our infrasound dataset; however, we felt this was the best option based on the data we did have available to us.  Below are examples of our data before and after our sound processing.

Before Applying Low-Pass Filter
After Applying Low-Pass Filter

Spectrograms are a visual representation of the spectrum of frequencies in a signal. The intensity of the frequencies in a signal are represented by color. The warmer colors (yellow), represent higher intensity at a given frequency while colder colors (blue) represent a lower intensity. From the spectrograms of one of our audio clips before applying a low pass filter, one can easily see the high intensity of the audio data at many frequencies between 0-8000 Hz. After applying the low pass filter to this same audio file, one can see how the intensity of our audio file is very high at low frequencies and very low for all higher frequencies. The original audio dataset was used for training our non-infrasound model and our new processed dataset was used for training our infrasound model.

## 10. Block B: M.L. Model Training

Figure 10.A: Block B 

The main purpose of Block B was the identification and training of a machine learning model capable of accurate real time detection of drones based on an audio input. 
### 10.1 Performance Requirements
Req #
Requirement
Description
B1
Audio Preprocessing
Must be able to preprocess one second segment of audio data by computing spectrogram to use as input to the ML model.
B2
Machine Learning Model
Model must accept a spectrogram of the audio clip and accurately predict whether the drone exists.


### 10.2 Input/Output Summary 
Type
Description
Notes
Input
Audio Clip from Stream
One second audio segments are the input to this block.
Output
Prediction
Returns a prediction of whether or not a drone is currently present.


### 10.3 Cost Requirements 
There are currently no costs associated with this block as all of the software used in this block is  free to use.
### 10.4 Environmental Requirements 
There are no environmental requirements associated with this block.
### 10.5 Machine Learning Algorithm Selection
Due to the complexity of trying to detect a drone based on its audio characteristics in real time, we wanted to try out various machine learning algorithms. We quickly decided we would use neural networks for our project because of their ability to improve without being explicitly programmed. Neural networks are able to find patterns in the data they are trained on, and once trained the machine learning model can be used to find patterns in data that the model has not seen before, or in our case determine if a drone is present or not in audio data input into the model.

From the neural network algorithms we tested, we tested them with validation data to determine which would be the most accurate without being too computationally expensive to be used in real-time detection.  Of the algorithms we tried, two showed the most potential for our project. The first was a fully connected neural network which was trained using the Short-Time Fourier Transform calculated for each audio file. The second was a convolutional neural network which was trained using the spectrograms created for each audio file.

After training the model, and testing them, as well as the recommendation from our mentor, we ultimately decided to use the convolutional neural network for our project.
### 10.6 Machine Learning Model Design
We created our machine learning model using TensorFlow. Our model consisted of 10 layers. The first layer was a rescaling layer to make the RGB image values more suitable for a neural network. (Note: While our model was using spectrograms for detection and associated color values, the model used the raw data and not an image as the input.) There were three convolutional layers with a max pool layer in each of them. On top of those is a fully connected layer that uses ‘ReLu’ as an activation function.
### 10.7 Machine Learning Model Training
To train our machine learning model, we used 16,000 audio files consisting of a random mix of drone and non-drone files from the dataset created in Block A. We split the data in the following way, 70 % of the data is used for training, 15% of the data is used for testing, and 15% of the data is used for validation of our model. Our models were trained using a batch size of 128.

Because we ultimately decided to extend our program to run in two real-time detection modes, infrasound and non-infrasound, we had to train two models for our project, one using the preprocessed infrasound data and one using the original unfiltered audio data.
### 10.8 Training Optimization - Epochs 
When training machine learning models, the accuracy of the models can be diminished due to overfitting. Overfitting refers to a trained model that corresponds to the data it was trained on too well and is a result of training the model using too many epochs [2]. The number of epochs used when training a model is the number of times the machine learning algorithm will work through the entire training dataset. The result of overfitting, is a model that has high accuracy when classifying data used to train it, but low accuracy when classifying new unseen data. 

When training models, one can determine if a model becomes overfit throughout a given number of epochs when the accuracy of the model on the training data increases, while the accuracy of the model on the validation data does not improve or even gets worse. Below is an example of what overfitting looks in practice and is a plot of the accuracy loss and validation loss of a large number of epochs.

Figure 10.B: Accuracy Loss vs. Validation Loss

In this example, overfitting begins to occur after roughly 10 epochs when validation loss plateaus and begins to increase while the accuracy loss continues to decrease.

To avoid overfitting of our model while ensuring the highest level of accuracy possible, we had to make sure it was not trained over too many epochs. However, we did not want to hardcode the number of epochs because we wanted to allow a user to retrain the model using any dataset they want and the number of elements in the dataset can vary. To accomplish this we used TensorFlow’s ‘patience’ variable that allows you to determine how many epochs can pass where validation accuracy doesn't improve before ending the training of the model. We decided to set the patience to 2 epochs to ensure the model cannot be trained more but avoiding any real decrease in validation accuracy.
### 10.9 Machine Learning Model Robustness
One objective of our project was to make sure our model is as accurate and robust as possible in order to ensure safety and security to its users. A model is considered robust if it produces consistent and correct classification on data it has not seen before. The robustness of a machine learning model can be improved in a variety of ways, the simplest being using a larger set of data for training. However, because infrasound data is not widely available and hard to come by, we felt this was not the best way to improve the robustness of our model for our client. To improve the robustness of our model, we experimented with data augmentation to slightly alter our existing data in different ways such as slightly adjusting the pitch and speed of our audio clips. This technique allows us to synthetically create a larger dataset to use when training our models.
### 10.10 Model Training Results
Our non-infrasound model trained over 10 epochs and achieved an accuracy of 98.3% and a validation accuracy of 97.5%. Our infrasound model also trained over 10 epochs and achieved an accuracy of 60.43% and a validation accuracy of 58.22%.

## 11. Block C: Real-Time Detection

Figure 11.A: Block C 

The main objective of Block C was the implementation of the trained machine learning model created in Block B into a function capable of real time detection. The output of this block is a prediction for every chunk of data determining if a drone is present or not.
### 11.1 Block Requirements
Req #
Requirement
Description
C1
Drone Detections
This block must accept incoming real-time audio and detection drones accurately using training machine learning models.

### 11.2 Input/Output Summary 
Type
Description
Notes
Input
Real-Time Audio
This block initializes an audio stream to accept real-time audio
Output
Prediction
Returns a prediction of whether or not a drone is currently present in audio input.


### 11.3 Cost Requirements 
There are no costs associated with this block as all of the software used was publicly available.
### 11.4 Environmental Requirements 
There are no environmental requirements associated with this block.
### 11.4 Audio Stream
We used PyAudio in this block for our audio stream. PyAudio provides Python bindings for PortAudio. Once initialized, the stream reads in one second chunks of audio data at a time which are used as the input to the machine learning model.
### 11.5 Real-Time Detection Results
When testing the accuracy of our real-time detection functionality we first used the simulated real-time audio data we created from Block A. This audio was played while running our real-time detection function. After playing the simulated real-time audio, we compared how many times a drone was detected to the number of seconds a drone was actually present in the audio. This technique works because our function takes in one second chunks of data at a time.
In this testing we achieved accuracies around 90%. 

## 12. Block D: Model Retraining 

Figure 12.A: Block D 


This main objective of Block D was the implementation of the machine learning model training code used in Block B into a function capable of training new machine learning models given a new dataset of the users choosing. This block was the second major addition to our project in order to provide a better product for our client.

We added this block because our program is unlikely to be as accurate when detecting based on infrasound as we would like given the data available. This block provides the functionality for our client to train a new machine learning model with true infrasound if and when it becomes available.

### 12.1 Block Requirements
Req #
Requirement
Description
D1
Model Retraining
- Train new machine learning model for any new dataset

### 12.2 Input/Output Summary 
Type
Description
Notes
Input 
Name 
- The name input is the name the retraining function will save the new model as.
Input
Path
- The path input is a path to wherever the new dataset is on the users computer
Output
New Model 
-The output of this block is a new machine learning block saved in the Models directory in our software directory.

### 12.3 Cost Requirements 
There are no costs associated with this block as all of the software used was publicly available.
### 12.4 Environmental Requirements 
There are no environmental requirements associated with this block.
### 12.5 Model Training 
This block accepts a path to a directory containing two labeled subdirectories and a new name. The model training function maintains the same model design and training characteristics mentioned previously in Block B. A new model can be trained on any dataset the user would like by accepting a path input. The directory containing the new dataset must contain two labeled subdirectories as labels for training are generated based on the names of their respective subdirectories. The newly trained models are saved into a model directory automatically with a name provided by the user.

## 13. Block E: User Interface

Figure 13.A: Block E 

This main objective of Block E was the combination of all previously mentioned functionality from our project. Block E is also responsible for ensuring users are easily able to interact with and control the functionality of our program.

For the user interface of our program, our minimum objective given to us by our client was a command line interface, as they are familiar with that structure and because this is not a commercial project that needs to be simple to use for everyone. However, we wanted to add the ability to control our program using a simple GUI to better serve our clients.

### 13.1 Block Requirements
Req #
Requirement
Description
E1
Complete
This block must combine all functionality of project
E2
User Friendly
The user must be able to easily interact with our program

### 13.2 Input/Output Summary 
Type
Description
Notes
Input 
Command Line Arguments
Command line arguments can be used to initiate all functionality associated with our project.
Input
GUI Inputs
When our program is run in GUI mode, there can be inputs for model retraining.

### 13.3 Cost Requirements 
There are no costs associated with this block as all of the software used was publicly available.
### 13.4 Environmental Requirements 
There are no environmental requirements associated with this block.
### 13.5 Program Packaging
This block is where we combined all functionality from our project into a single Python program that can be controlled by running it one of two modes - using command line arguments or a graphical user interface. The graphical user interface is the default running mode to simplify use for users not familiar with command line arguments. Running the program with both command line arguments or the GUI allow for the same functionality. The different functions that our program can handle are further explained in the following sections.

### 13.6 Software Flow Chart

Figure 13.B: Software Flow Chart 
When running our program, the user can control it in one of two ways - a graphical user interface or command line arguments. To run our program with a graphical user interface, the user would run our program with no arguments and the GUI would launch automatically. The program would then be controlled using the GUI. If the user chooses to use command line arguments, the program can be run with an argument to launch one of two modes - one for real-time detection or one for model retraining. An argument must also be passed to specify the type of sound the mode is to be run on: infrasound or non-infrasound.
### 13.7 Infrasound vs. Non-Infrasound
Our client asked for software capable of drone detection based on infrasound. However, because the dataset we are using is not true infrasound, we decided to allow our use to select the audio data type when running our program for a couple reasons:
The result of filtering our data to extract the infrasound created noise, as the microphones the data was originally recorded on are not designed for such low frequencies. As a result, our program is unlikely to be as accurate when detecting based on infrasound as we would like.
Infrasound is not audible to humans, we wanted to be able to better demonstrate our programs capabilities with drone audio data we can actually hear.

When running our program, the user will be able to choose between the two sound types, and both model training and real-time detection capabilities are available for both.
### 13.8 Command Line Arguments 
When running our program using command line arguments, the following arguments are available showing in [Fig. 13.C].

Figure 13.C: Available Command Line Arguments
Below are example commands that could be used to run our program:
Model Training (Normal)
$ python drone_detection.py -r "~/Desktop/Capstone/E490-Capstone/<data_directory>” -n "test_train_normal"

Model Training (Infrasound)
$ python drone_detection.py -i  -r "~/Desktop/Capstone/E490-Capstone/<data_directory>” -n "test_train_infrasound"

Real-Time (Normal)
$ python drone_detection.py -d "normal_RealTime_Default"

Real-Time (Infrasound)
$ python drone_detection.py -i -d "infrasound_RealTime_Default"


### 13.9 Graphical User Interface

Figure 13.D: Graphical User Interface

Above in [Fig. 13.D] is what our GUI looks like once launched. Our GUI is broken down into three sections: Real Time Detection, Model Retraining, and Detection Logs. In the bottom left of the GUI is a button to pull up a help menu with instructions on how to control and use our program shown below in [Fig 13.D.2].

Figure 13.D.2: Help Menu
#### 13.9.1 Real Time Detection Mode

Figure 13.E: Real-Time Detection Section



Figure 13.F: Real-Time Running In Terminal 

Above shown in [Fig. 13.E]  is the real-time detection section, this contains a list of all models that have been trained previously contained in our model directory. All models are saved with the sound type it was trained for to allow users to easily tell the difference between infrasound and normal models. When the user would like to start real-time detection for a given model, they can simply select the model they would like to use and real-time detection will automatically start. If the user would like to watch for detections in real-time, that is shown in the termina. The image [Fig 13.F] shows what the terminal window looks like when running real-time detection mode is running. When drones are detected, alerts are printed to the terminal and an event is added to a logging file with the time of detection.
#### 13.9.2 Model Retraining Mode
   
Figure 13.G: Model Training Example

The image shown above in [Fig 13.G] shows an example of the inputs used to train a new model. The user must provide a path to the dataset and a name for the model to be saved as. When model training is finished, there is a pop-up window in the GUI notifying the user that training is complete. Refreshing the models list in the real-time detection section will show the newly trained model which can then be used to launch a real-time detection session.

#### 13.9.3 Detections Logs

Figure 13.G: Detection Logs Section

Figure 13.H: Example Detection Log 

Above shown in [Fig. 13.G] is the detection logs section, this contains a list of all of the detections created during previous real-time detection sections. Each time a real-time detection section is launched, a new log file is created and the name is the date and time the session started as well as the sound type. The image in [Fig 13.H] shows the formatting and output included in an example log. Each time a drone is detected a new log event is added with the time of the detection. 

## 14. Future Interests
While our work on this project may be officially over soon, going forward this project could benefit by the exploration and of generating infrasound using a signal generator and using this artificial data for training and testing of the machine learning model as well as the input to the real-time detection functionality. The data we used was not recorded on a microphone designed to detect low frequency infrasound, because of this the data at these frequencies is noisy after sound processing. Also, the drone data we used was for consumer drones rather than military drones, which currently inhibits the use of our product by our client for its intended purposes. We believe a better dataset would add significant value to the project by increasing the accuracy of our machine learning model.

## 15. Future Plans 
### 15.1 Ethan
Throughout the capstone project and my time as an ISE undergraduate, I have learned many valuable skills that have helped me decide what my career interests are. These interests include embedded software development and FPGA development and the applications where the two may intersect.  The capstone project specifically has taught me a lot about the value of working within a team and how proper communication and planning can be one of the most important aspects for bringing a project to completion. As for my post college plans, I have accepted a position as a Software Engineer at ARCTEC Solutions, which is a defense contractor startup located in Northern Virginia working mainly on Software Defined Radio applications.
### 15.2 Brian 
During our capstone project, I learned a lot of new things and refined many of the skills I have learned throughout my time in the ISE program. Over the past four years I have been grateful for the opportunity to pursue my interests in computing, machine learning, embedded systems, and more.  Next year I will be working towards my Masters in ISE and intend to complete that in May 2022. This summer I will be staying in Bloomington and working as an undergraduate instructor as well as completing an independent study with Professor Himebaugh.
### 15.3 Andrew
Throughout this project and degree, I learned many skills that have helped me define my career interest.  While the career path I thought I wanted to go into has certainly evolved over the course of my undergraduate experience, I have decided to focus more on integration and software engineering. Throughout 2020 I worked as an integrations engineering intern at Cray supercomputing, and this summer I will continue with that work.  After this role is finished I will be returning to the ISE program to pursue my masters degree in Computer Engineering.

## 16. Acknowledgments 
We would like to thank Indiana University and the Luddy School of Informatics, Computing, and Engineering for the opportunity to work on a Capstone Project. We would like to thank Professor Suha Lasassmeh for the guidance and assistance throughout the year and the time she has spent meeting with our group and Professor Bryce Himebaugh for his help during the second semester. We would also like to thank our industry sponsor, the United States Navy NSWC Crane, for the opportunity to complete a project for them and would like to thank our Mentor Ryan Dowd for the guidance and resources he has provided, and the time he has taken to meet with our group.

## 17. References
[1]: Al-Emadi, Sara & Al-Ali, Abdulla & Mohammed, Amr & Al-Ali, Abdulaziz. (2019). Audio Based Drone Detection and Identification using Deep Learning. 10.1109/IWCMC.2019.8766732.

[2] Wikipedia contributors. (2021, April 8). Overfitting. In Wikipedia, The Free Encyclopedia. Retrieved 18:07, April 11, 2021, from https://en.wikipedia.org/w/index.php?title=Overfitting&oldid=1016721642

[3] S. Al-Emadi, Saraalemadi/droneaudiodataset, 2018.[Online]. 
Available: https://github.com/saraalemadi/DroneAudioDataset.

