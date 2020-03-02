# USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib

import imutils
from keras.preprocessing.image import img_to_array

matplotlib.use("Agg")

from CNN.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from keras.optimizers import Adam
import time
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
import glob

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from sklearn.metrics import confusion_matrix


#%%
# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 10
EPOCHS = 25
# Define the Keras TensorBoard callback.
NAME = "Live vs Fake photos" + str(int(time.time()))
tensorboard_callback = TensorBoard(log_dir="logs\\{}".format(NAME))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_images_raw() :
	print("[INFO] loading images...")
	data = []
	labels = []
	labels1 = ["ImposterRaw", "ClientRaw"]
	for label_name in labels1 :
		print('Doing label: ', label_name)
		for imagePath in glob.iglob(f'dataset/raw/{label_name}/*/*.jpg') :
			print(imagePath)
			# extract the class label from the filename, load the image and
			# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
			image = cv2.imread(imagePath)
			frame = imutils.resize(image, width=600)
			faces = faceCascade.detectMultiScale(
					image,
					scaleFactor=1.3,
					minNeighbors=3,
					minSize=(30, 30)
			)
			for (x, y, w, h) in faces :
				roi_color = frame[y :y + h, x :x + w]
				face = frame[y :y + h, x :x + w]
				face = cv2.resize(face, (32, 32))
				face = face.astype("float") / 255.0
			
			# update the data_features and labels lists, respectively
			data.append(face)
			if (label_name == 'ImposterRaw') :
				labels.append(0)
			else :
				labels.append(1)
				
	return data,labels


def get_images_detected() :
	print("[INFO] loading images...")
	data = []
	labels = []
	labels1 = ["ImposterFace", "ClientFace"]
	for label_name in labels1:
		print('Doing label: ' , label_name)
		for imagePath in glob.iglob(f'dataset/face_no_mouth/{label_name}/*/*.jpg'):
			print(imagePath)
			# extract the class label from the filename, load the image and
			# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
			image = cv2.imread(imagePath)
			
			# print(imagePath)
			image = cv2.resize(image, (32, 32))
			
			# update the data_features and labels lists, respectively
			data.append(image)
			if (label_name == 'ImposterFace'):
				labels.append(0)
			else:
				labels.append(1)
				
	return data,labels



data,labels = get_images_raw()
# data,labels = get_images_detected()


# Detected Face 20 epochs:
"""
{'test_detectedface':  0.757009
'test_face_both_eyes':  0.640494
'test_face_no_left_eye':  0.698462
'test_face_no_right_eye':  0.662857
'test_face_no_mouth':  0.46989
'test_face_no_nose':  0.468132}
"""
# Detected Face 25 epochs:
"""
{'test_detectedface':  0.903144
'test_face_both_eyes':  0.660785
'test_face_no_left_eye':  0.814945
'test_face_no_right_eye':  0.745055
'test_face_no_mouth':  0.468132
'test_face_no_nose':  0.469011
"""
# raw Face 20 epochs:
"""
{'test_detectedface':  0.825404
'test_face_both_eyes':  0.535509
'test_face_no_left_eye':  0.619341
'test_face_no_right_eye':  0.607033
'test_face_no_mouth':  0.469011
'test_face_no_nose':  0.469011
"""
# raw Face 25 epochs:
"""
{'test_detectedface':  0.895497
'test_face_both_eyes':  0.856198
'test_face_no_left_eye':  0.891868
'test_face_no_right_eye':  0.876923
'test_face_no_mouth':  0.810989
'test_face_no_nose':  0.674286
"""
# face no mouth 20 epochs:
"""
{'test_detectedface':  0.933305
dtype: float64, 'test_face_both_eyes':  0.929863
dtype: float64, 'test_face_no_left_eye':  0.95956
dtype: float64, 'test_face_no_right_eye':  0.93978
dtype: float64, 'test_face_no_mouth':  0.754286
dtype: float64, 'test_face_no_nose':  0.745934
"""
# face no mouth 25 epochs:
"""
{'test_detectedface': 0.975786
'test_face_both_eyes': 0.854433
'test_face_no_left_eye': 0.94022
'test_face_no_right_eye': 0.949451
'test_face_no_mouth': 0.925714
'test_face_no_nose': 0.781978
"""

#%%
print(len(data))
#%% 
# convert the data_features into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)



# training data_features ( trainX ) and training labels ( trainY ).
(trainX, testX, trainY, testY) = train_test_split(data, labels,
												  test_size=0.20, random_state=42)


	
# apply data_features augmentation, randomly translating, rotating, resizing, etc. images on the fly.
# enabling our model to generalize better
aug = ImageDataGenerator( rescale = 1./255,
                                   shear_range = 0.2,
								   width_shift_range=0.2, 
								   height_shift_range=0.2,
								   rotation_range=90,
								   brightness_range=[0.2,1.0],
								   zoom_range=[0.5,1.0],
								   featurewise_center=True,
								  featurewise_std_normalization=True,
                                   horizontal_flip = True,
								    fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
						  classes=len(le.classes_))

# np.savetxt('feature_extraction/features_no_left_eye.csv', model.predict(trainX, batch_size=BS), delimiter=',')
# np.savetxt('feature_extraction/labels_no_left_eye.csv', trainY, delimiter=',')
#
# np.savetxt('feature_extraction/features_no_left_eye.txt', model.predict(trainX, batch_size=BS), delimiter=',')
# np.savetxt('feature_extraction/labels_no_left_eye.txt', trainY, delimiter=',')

model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(len(le.classes_)))
model.add(Activation("softmax"))

model.compile(loss="binary_crossentropy", optimizer=opt,
			  metrics=["accuracy"])

model.summary()
# model.save_weights('my_weights.h5')

#%% 

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
# fit generator is on infinite look so do steps per epoch to terminate it
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
						validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
						epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
# print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network to '{}'...".format('glasses_model.h5'))
# model.save('liveness.model')
model.save('NUAA_dataset_final.h5')

# save the label encoder to disk
f = open('NUAA_dataset_final.pickle', "wb")
f.write(pickle.dumps(le))
f.close()
#%%


dict = {}
labels_t = ["ImposterRaw", "ClientRaw"]
for label_name in labels_t:
	print('Doing label: ' , label_name)
	for imagePath in glob.iglob(f'dataset/test_raw/{label_name}/*/*.jpg'):
			print(imagePath)
			image = cv2.imread(imagePath)
			frame = imutils.resize(image, width=600)
			faces = faceCascade.detectMultiScale(
			image,
			scaleFactor=1.3,
			minNeighbors=3,
			minSize=(30, 30)
			)
			for (x, y, w, h) in faces :
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				#get pixel locations of the box to extract face
				roi_color = frame[y :y + h, x :x + w]
				face = frame[y :y + h, x :x + w]
				face = cv2.resize(face, (32, 32))
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)
		
				preds = model.predict(face)[0]
				j = np.argmax(preds)
				label = le.classes_[j]
				if (label == 1):
					label='real'
				else:
					label = 'fake'
			dict[imagePath] = label
	
print(dict)
import pandas as pd
df = pd.DataFrame(list(dict.items()), columns=['Filename', 'Prediction'])
df.to_csv("raw.csv",index=False)

dict1 = {}
labels_t = ["ImposterFace", "ClientFace"]
for label_name in labels_t:
	print('Doing label: ' , label_name)
	for imagePath in glob.iglob(f'dataset/test_detectedface/{label_name}/*/*.jpg'):
			print(imagePath)
			image = cv2.imread(imagePath)
			frame = imutils.resize(image, width=600)
			frame = cv2.resize(frame, (32, 32))
			frame = frame.astype("float") / 255.0
			frame = img_to_array(frame)
			frame = np.expand_dims(frame, axis=0)
			preds = model.predict(frame)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
			if (label == 1):
				label='real'
				
			else:
				label = 'fake'
			dict1[imagePath] = label
	
print(dict1)
df = pd.DataFrame(list(dict1.items()), columns=['Filename', 'Prediction'])
df.to_csv("detected.csv",index=False)



actual = model.predict(testX)
actual = np.argmax(actual, axis=1) # axis 1 = rows, axis 0 = columns

""" argmax returns the index of the maximum value in each of the rows in the model"""
results = confusion_matrix(np.argmax(testY, axis=1), actual)
report_1 = classification_report(np.argmax(testY, axis=1), actual, target_names=['actual' , 'expected'])
print("conf: " ,results)
print("report_1: " ,report_1)



# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')

score = model.evaluate_generator(aug.flow(trainX, trainY, batch_size=BS), verbose=1, steps=500)
print("Metric Names are : ", model.metrics_names)  # ['loss', 'accuracy']
print("Final Accuracy is: " + str(score))
print("Shape trainX", trainX.shape)
print("Shape testX", testX.shape)
print("Shape trainY", trainY.shape)
print("Shape testY", testY.shape)
#%% 



