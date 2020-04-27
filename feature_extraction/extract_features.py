# set the matplotlib backend so figures can be saved in the background
import matplotlib



matplotlib.use("Agg")

from network import build
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import time
from keras.callbacks import TensorBoard
import glob
from os import path
from sklearn.preprocessing import OneHotEncoder
#%%
# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 10
EPOCHS = 60
# Define the Keras TensorBoard callback.
NAME = "Live vs Fake photos" + str(int(time.time()))
tensorboard_callback = TensorBoard(log_dir="logs\\{}".format(NAME))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

common_img_paths = []

def get_org_images_detected(single_dataset) :
	print("[INFO] loading images...")
	data = []
	labels = []
	labels1 = ["ImposterFace", "ClientFace"]
	for label_name in labels1:
		print('Doing label: ' , label_name)
		for imagePath in glob.iglob(f'../dataset/{single_dataset}/{label_name}/*/*.jpg'):
			if (single_dataset == 'Detectedface'):
				if (path.exists(imagePath.replace("Detectedface", "face_both_eyes")) and path.exists(imagePath.replace("Detectedface", "face_no_left_eye")) and path.exists(imagePath.replace("Detectedface", "face_no_mouth")) and path.exists(imagePath.replace("Detectedface", "face_no_nose")) and path.exists(imagePath.replace("Detectedface", "face_no_right_eye"))):
					print(imagePath)
					common_img_paths.append(imagePath)
					# extract the class label from the filename, load the image and
					# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
					image = cv2.imread(imagePath)
					
					# print(imagePath)
					image = cv2.resize(image, (32, 32))
					
					# update the data_features and labels lists, respectively
					data.append(image)
					if (label_name == 'ImposterFace'):
						labels.append([0])
					else:
						labels.append([1])
				
	return data,labels

def get_images_detected(single_dataset) :
	print("[INFO] loading images...")
	data = []
	labels = []
	labels1 = ["ImposterFace", "ClientFace"]
	for label_name in labels1:
		print('Doing label: ' , label_name)
		for imagePath in glob.iglob(f'../dataset/{single_dataset}/{label_name}/*/*.jpg'):
			if (imagePath.replace(f"{single_dataset}", "Detectedface") in common_img_paths):
				print(imagePath)
				# extract the class label from the filename, load the image and
				# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
				image = cv2.imread(imagePath)
				
				# print(imagePath)
				image = cv2.resize(image, (32, 32))
				
				# update the data_features and labels lists, respectively
				data.append(image)
				if (label_name == 'ImposterFace'):
					labels.append([0])
				else:
					labels.append([1])
				
	return data,labels

datasets = ['Detectedface','face_both_eyes','face_no_left_eye','face_no_mouth','face_no_nose','face_no_right_eye']


data,labels = get_org_images_detected(datasets[0])
data1,labels1 = get_images_detected(datasets[1])
data2,labels2 = get_images_detected(datasets[2])
data3,labels3 = get_images_detected(datasets[3])
data4,labels4 = get_images_detected(datasets[4])
data5,labels5 = get_images_detected(datasets[5])

# print(len(common_img_paths))
# print(len(data))
# print(len(data1))

data = np.array(data, dtype="float") / 255.0
data1 = np.array(data1, dtype="float") / 255.0
data2 = np.array(data2, dtype="float") / 255.0
data3 = np.array(data3, dtype="float") / 255.0
data4 = np.array(data4, dtype="float") / 255.0
data5 = np.array(data5, dtype="float") / 255.0

enc = OneHotEncoder()
enc.fit(labels)
enc.fit(labels1)
enc.fit(labels2)
enc.fit(labels3)
enc.fit(labels4)
enc.fit(labels5)
labels_n = enc.transform(labels).toarray()
labels_n1= enc.transform(labels1).toarray()
labels_n2= enc.transform(labels2).toarray()
labels_n3= enc.transform(labels3).toarray()
labels_n4= enc.transform(labels4).toarray()
labels_n5= enc.transform(labels5).toarray()




(trainX, testX, trainY, testY) = train_test_split(data, labels_n,test_size=0.20, random_state=42)
(trainX1, testX1, trainY1, testY1) = train_test_split(data1, labels_n1,test_size=0.20, random_state=42)
(trainX2, testX2, trainY2, testY2) = train_test_split(data2, labels_n2,test_size=0.20, random_state=42)
(trainX3, testX3, trainY3, testY3) = train_test_split(data3, labels_n3,test_size=0.20, random_state=42)
(trainX4, testX4, trainY4, testY4) = train_test_split(data4, labels_n4,test_size=0.20, random_state=42)
(trainX5, testX5, trainY5, testY5) = train_test_split(data5, labels_n5,test_size=0.20, random_state=42)


print("[INFO] compiling model...")
model = build(width=32, height=32, depth=3,
						  classes=2)
np.savetxt(f'features_Detectedface.csv', model.predict(trainX, batch_size=BS), delimiter=',')
np.savetxt(f'labels_Detectedface.csv', trainY, delimiter=',')

np.savetxt(f'features_face_both_eyes.csv', model.predict(trainX1, batch_size=BS), delimiter=',')
np.savetxt(f'labels_face_both_eyes.csv', trainY1, delimiter=',')

np.savetxt(f'features_face_no_left_eye.csv', model.predict(trainX2, batch_size=BS), delimiter=',')
np.savetxt(f'labels_face_no_left_eye.csv', trainY2, delimiter=',')

np.savetxt(f'features_face_no_mouth.csv', model.predict(trainX3, batch_size=BS), delimiter=',')
np.savetxt(f'labels_face_no_mouth.csv', trainY3, delimiter=',')

np.savetxt(f'features_face_no_nose.csv', model.predict(trainX4, batch_size=BS), delimiter=',')
np.savetxt(f'labels_face_no_nose.csv', trainY4, delimiter=',')

np.savetxt(f'features_face_no_right_eye.csv', model.predict(trainX5, batch_size=BS), delimiter=',')
np.savetxt(f'labels_face_no_right_eye.csv', trainY5, delimiter=',')
