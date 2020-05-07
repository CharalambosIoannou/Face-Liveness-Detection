"""
To run file: type in a terminal python evaluations_on_model.py

"""
import imutils
import matplotlib



matplotlib.use("Agg")
from sklearn.preprocessing import OneHotEncoder
from network import build
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import glob

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef,classification_report
import pandas as pd
import time
#%%

INIT_LR = 1e-4
BS = 10
# EPOCHS = 5
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
		for imagePath in glob.iglob(f'../dataset/raw/{label_name}/*/*.jpg') :
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
				roi_color = frame[y :y + h, x :x + w]
				face = frame[y :y + h, x :x + w]
				face = cv2.resize(face, (32, 32))
				face = face.astype("float") / 255.0
			
			data.append(face)
			if (label_name == 'ImposterRaw') :
				labels.append([0])
			else :
				labels.append([1])
				
	return data,labels


def get_images_detected() :
	print("[INFO] loading images...")
	data = []
	labels = []
	labels1 = ["ImposterFace", "ClientFace"]
	for label_name in labels1:
		print('Doing label: ' , label_name)
		for imagePath in glob.iglob(f'../dataset/Detectedface/{label_name}/*/*.jpg'):
			print(imagePath)
	
			image = cv2.imread(imagePath)
			
			image = cv2.resize(image, (32, 32))
			
			data.append(image)
			if (label_name == 'ImposterFace'):
				labels.append([0])
			else:
				labels.append([1])
				
	return data,labels

epochs = [5,10,25,40,50,60,85,100,150,200]
# epochs = [1,2,3]

dfObj = pd.DataFrame(columns=['epoch', 'Accuracy', 'Precision', 'Recall/TPR', 'f1', 'Kappa', 'Matt', 'TNR', 'FPR','train_time'])
for EPOCHS in epochs:
	# data,labels = get_images_raw()
	data,labels = get_images_detected()
	
	#%%
	print(len(data))
	#%%

	data = np.array(data, dtype="float") / 255.0


	enc = OneHotEncoder()
	enc.fit(labels)
	labels = enc.transform(labels).toarray()
	
	
	#%%
	# training data_features ( trainX ) and training labels ( trainY ).
	(trainX, testX, trainY, testY) = train_test_split(data, labels,
													  test_size=0.20, random_state=42)
	
	
		
	
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
	
	#%%
	
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model = build(width=32, height=32, depth=3,
							  classes=2)

	
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	
	model.add(Dense(2))
	model.add(Activation("softmax"))
	
	model.compile(loss="categorical_crossentropy", optimizer=opt,
				  metrics=["accuracy"])
	
	model.summary()

	
	#%%
	
	print("[INFO] training network for {} epochs...".format(EPOCHS))
	start = time.time()
	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
							validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
							epochs=EPOCHS)
	finish = time.time()
	
	
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=BS)
	
	
	y_pred = np.argmax(predictions, axis=1)
	y_test = np.argmax(testY, axis=1)
	""" argmax returns the index of the maximum value in each of the rows in the model"""
	results = confusion_matrix(y_test, y_pred)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	report_1 = classification_report(y_test,y_pred , target_names=['actual' , 'expected'])
	# Precision
	pr = precision_score(y_test, y_pred)
	
	# Recall
	re = recall_score(y_test, y_pred)
	
	# F1 score
	f1 = f1_score(y_test,y_pred)
	
	# Cohen's kappa
	co = cohen_kappa_score(y_test, y_pred)
	
	# matthews_corrcoef
	ma = matthews_corrcoef(y_test, y_pred)
	
	acc = (tp+tn) / (tp+tn+fp+fn)
	tnr =tn / (fp+tn)
	fpr = fp / (fp+tn)
	train_time = finish - start
	print("conf: " ,results)
	print("report_1: " ,report_1)
	
	print("tn: " ,tn)
	print("fp: " ,fp)
	print("fn: " ,fn)
	print("tp: " ,tp)
	print("pr: " ,pr)
	print("TPR, re: " ,re)
	print("f1: " ,f1)
	print("co: " ,co)
	print("ma: " ,ma)
	print("acc: " ,acc)
	print("TNR: " ,tnr)
	print("FPR: " ,fpr)
	dfObj = dfObj.append({'epoch': EPOCHS, 'Accuracy': acc, 'Precision': pr, 'Recall/TPR': re, 'f1':f1, 'Kappa':co, 'Matt':ma,'TNR':tnr, 'FPR':fpr,'train_time':train_time}, ignore_index=True)
	
	print(dfObj)
	dfObj.to_csv('evaluations.csv')
dfObj.to_csv('evaluations.csv')
#%%



