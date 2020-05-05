
import imutils
import matplotlib
matplotlib.use("Agg")
from network import build
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from keras.optimizers import Adam
import time
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
import glob
import keras
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, \
	roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
#%%
INIT_LR = 1e-4
BS = 10
EPOCHS = 60
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
# data,labels = get_images_raw()
data,labels = get_images_detected()

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

# np.savetxt('../feature_extraction/features_org.csv', model.predict(trainX, batch_size=BS), delimiter=',')
# np.savetxt('../feature_extraction/labels_org.csv', trainY, delimiter=',')

# np.savetxt('../feature_extraction/features_no_both_eyes.txt', model.predict(trainX, batch_size=BS), delimiter=',')
# np.savetxt('../feature_extraction/labels_no_both_eyes.txt', trainY, delimiter=',')

model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dropout(0.5))


# softmax classifier
model.add(keras.layers.core.Dense(2))
model.add(keras.layers.core.Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer=opt,
			  metrics=["accuracy"])

model.summary()
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#%%
# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
# fit generator is on infinite look so do steps per epoch to terminate it
augm = aug.flow(trainX, trainY, batch_size=BS)
H = model.fit_generator(augm,
						validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
						epochs=EPOCHS)



print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)

print("[INFO] exporting to model_save.h5")
model.save('model_save.h5')
f = open('model_save.pickle', "wb")
f.write(pickle.dumps(enc))
f.close()
#%%


# dict = {}
# labels_t = ["ImposterRaw", "ClientRaw"]
# for label_name in labels_t:
# 	print('Doing label: ' , label_name)
# 	for imagePath in glob.iglob(f'../dataset/test_raw/{label_name}/*/*.jpg'):
# 			print(imagePath)
# 			image = cv2.imread(imagePath)
# 			frame = imutils.resize(image, width=600)
# 			faces = faceCascade.detectMultiScale(
# 			image,
# 			scaleFactor=1.3,
# 			minNeighbors=3,
# 			minSize=(30, 30)
# 			)
# 			for (x, y, w, h) in faces :
# 				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 				#get pixel locations of the box to extract face
# 				roi_color = frame[y :y + h, x :x + w]
# 				face = frame[y :y + h, x :x + w]
# 				face = cv2.resize(face, (32, 32))
# 				face = img_to_array(face.astype("float") / 255.0)
# 				face = np.expand_dims(face, axis=0)
# 				preds = model.predict(face)[0]
# 				j = np.argmax(preds)
# 				label = [0, 1][j]
# 				if (label == 1):
# 					label='real'
# 				else:
# 					label = 'fake'
# 			dict[imagePath] = label
#
# print(dict)
# import pandas as pd
# df = pd.DataFrame(list(dict.items()), columns=['Filename', 'Prediction'])
# df.to_csv("raw.csv",index=False)
#
# dict1 = {}
# labels_t = ["ImposterFace", "ClientFace"]
# for label_name in labels_t:
# 	print('Doing label: ' , label_name)
# 	for imagePath in glob.iglob(f'../dataset/test_detectedface/{label_name}/*/*.jpg'):
# 			print(imagePath)
# 			image = cv2.imread(imagePath)
# 			frame = imutils.resize(image, width=600)
# 			frame = cv2.resize(frame, (32, 32))
# 			frame = img_to_array(frame.astype("float") / 255.0)
# 			frame = np.expand_dims(frame, axis=0)
# 			preds = model.predict(frame)[0]
# 			j = np.argmax(preds)
# 			label = [0,1][j]
# 			if (label == 1):
# 				label='real'
#
# 			else:
# 				label = 'fake'
# 			dict1[imagePath] = label
#
# print(dict1)
# df = pd.DataFrame(list(dict1.items()), columns=['Filename', 'Prediction'])
# df.to_csv("detected.csv",index=False)
#


actual = model.predict(testX)
actual = np.argmax(actual, axis=1) # axis 1 = rows, axis 0 = columns
print("after: " , actual)
print(np.argmax(testY, axis=1))

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

# Compute ROC curve and ROC area for each class


fpr = {}
tpr = {}
roc_auc = {}
for i in range(2):
	fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
	roc_auc[i] = auc(fpr[i], tpr[i])

# print (roc_auc_score(y_test, y_pred))
# plt.figure()
# plt.plot(fpr[1], tpr[1])
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.savefig('roc.png')
# plt.show()

plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc.png')


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epochs")
plt.ylabel("Accuracy and Loss")
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



