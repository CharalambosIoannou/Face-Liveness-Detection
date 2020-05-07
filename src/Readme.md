# **Scripts:** 

## evaluation_on_real_time.py
This file evaluates real time liveness detection when run

## evaluation_on_model.py
This file evaluates the model used to perform liveness detection when run

## extract_features.py
This file extracts the features from the original and occluded datasets and saves the features and the labels in csv files. The csv files are saved in the feature_extraction folder.

## face_occlusion.py
This file creates the occluded datasets from the original dataset. You specify which feature to occlude and then it produces the occluded dataset.

## gui.py
 This file is the graphical user interface of the project. It has two main functions. Run real time liveness detection or perform liveness detection for an input image

 ## histogram_subtraction.py
 This file reads all the csv files created from `extract_features.py` and it creates:
* 5 histograms showing real and fake features from the occluded datasets  
* 5 histograms showing the absolute differences between each feature
* One bar chart showing the sum of absolute differences

All the figures are saved in the feature_extraction/imgs subfolder.

##  liveness_from_photos.py
 This file performs liveness detection for an input image

##  liveness_from_video.py
 This file performs real time liveness detection using the laptop's camera

## model_save.h5
The saved model used to perform real time liveness detection fast

## model_save.pickle
The saved labels used to perform real time liveness detection fast

## network.py
The DCNN model used for liveness detection

## process_data.py
This file creates a histogram of the real and fake features from the clear (not occluded) dataset

## run.bat
Bat file used to run the program on Windows

## run.sh
Bash file used to run the program on Linux

## train_network.py
The code to train the network and obtain evaluation metrics


