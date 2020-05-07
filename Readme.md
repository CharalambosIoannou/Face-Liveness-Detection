# Facial Liveness Detection and identification of the most important feature

 3<sup>rd</sup> year project for BSc Computer Science at Durham University. Supervised by Dr Yang Long.

 ## Introduction

 ----

 The aim of this project is to implement a facial liveness detection algorithm using a deep learning model. In addition use the deep learning model to extract the facial features from the input dataset and identify the most imporant feature in the face that helps the algorithm distinguish which face is real and which is not. 

 This algorithm can also work in real time.

 ## Dependencies
 ----
 * Python 3.5+
 * imutils==0.5.3
 * numpy==1.17.2
 * matplotlib==3.1.1
 * tensorflow==2.1.0
 * pandas==0.25.2
 * Keras==2.3.1
 * opencv_python==4.1.1.26
 * dlib==19.19.0
 * scikit_learn==0.22.2.post1

 ## Folder structure
 ---
### **Folders:**

 ### dataset/
 This folder should contain all of the datasets used for this project. It has a doogle drive link in order to download the datasets.

### feature_extraction/
 This folder contains the features extracted from the original and occluded datasets. It also contains a subfolder named "imgs" which contains images of histograms showing the difference between the different facial features.  

### images/
This folder contains images used for the project.

 ### src/
 This folder contains all the code to run the system.






## How to run
1) Install required dependencies using the command `pip install --user -r requirements.txt`. If this fails for some reason then do `pip install <library name>` on all the libraries mentioned above one by one.
2) Navigate to src folder and click on `run.bat` file
3) When the graphical user inteface opens there are two options:
    * Perform real-time liveness detection which is going to open the laptop camera
    * Perform liveness detection from an image you want to choose
4) Press one of the buttons according to which liveness detection you want to carry out


### Note:

If the pip installation of Dlib outputs an error requiring CMake then first do `pip install cmake` and then do `pip install dlib`

 

 

