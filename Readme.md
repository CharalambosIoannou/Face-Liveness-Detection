# Facial Liveness Detection and identification of the most important feature

 3<sup>rd</sup> year project for BSc Computer Science at Durham University. Supervised by Dr Yang Long.

 ## Introduction

 ----

 The aim of this project is to implement a facial liveness detection algorithm using a deep learning model. In addition use the deep learning model to extract the facial features from the input dataset and identify the most imporant feature in the face that helps the algorithm distinguish which face is real and which is not. 

 This algorithm can also work in real time.

 ## Dependencies
 ----
 * Python 3.5+
 * Tensorflow

 ## Folder and file structure
 ---
### **Folders:**
 ### face_occlusion/
 This folder contains a script which iterates through the dataset and occludes the specified facial feature. This outputs a folder with identical structure and data as the original dataset but with the occluded images. 
 
 It also contains the file named "shape_predictor_68_face_landmarks.dat" which is used for face localization.

 ### dataset/
 This folder contains all of the datasets used for this project. It has a doogle drive link in order to download the datasets.

### feature_extraction/
 This folder contains the code to compare the occluded dataset with the original dataset. It also contains code to produce histograms of the orignal and occluded datasets

### **Files:** 
### gui.py
 This file is the graphical user interface of the project. It has two main functions. Run real time liveness detection or perform liveness detection for an input image

###  liveness_from_photos.py
 This file performs liveness detection for an input image

###  liveness_from_video.py
 This file performs real time liveness detection using the laptop's camera

### model_save.h5
The saved model used to perform real time liveness detection fast

### model_save.pickle
The saved labels used to perform real time liveness detection fast

### network.py
The DCNN model used for liveness detection

### train_network.py
The code to train the network and obtain evaluation metrics

### build_exe.bat
Bat file used to run the program on Windows

## How to run
Double click on "build_exe.bat" file when using Windows OS

 

 

