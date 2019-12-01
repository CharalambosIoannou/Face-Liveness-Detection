# USAGE
# python train_liveness.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib



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

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 10
EPOCHS = 1
# Define the Keras TensorBoard callback.
NAME = "Live vs Fake photos" + str(int(time.time()))
tensorboard_callback = TensorBoard(log_dir="logs\\{}".format(NAME))

# grab the list of images in our dataset directory, then initialize
# the list of data_features (i.e., images) and class images
print("[INFO] loading images...")
data = []
labels = []
labels1 = ["ImposterFace", "ClientFace"]
for label_name in labels1:
	print('Doing label: ' , label_name)
	for imagePath in glob.iglob(f'dataset/Detectedface/{label_name}/*/*.jpg'):
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

print(len(data))
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
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
						 width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
						 horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
						  classes=len(le.classes_))

# np.savetxt('feature_extraction/features.csv', model.predict(trainX, batch_size=BS), delimiter=',')
# np.savetxt('feature_extraction/labels.csv', trainY, delimiter=',')
#
# np.savetxt('feature_extraction/features.txt', model.predict(trainX, batch_size=BS), delimiter=',')
# np.savetxt('feature_extraction/labels.txt', trainY, delimiter=',')

model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(len(le.classes_)))
model.add(Activation("softmax"))

model.compile(loss="binary_crossentropy", optimizer=opt,
			  metrics=["accuracy"])



# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
# fit generator is on infinite look so do steps per epoch to terminate it
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
						validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
						epochs=EPOCHS, callbacks=[tensorboard_callback])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS) #TODO look at classification report and dynamic evaluation
# print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network to '{}'...".format('glasses_model.h5'))
# model.save('liveness.model')
model.save('NUAA_dataset.h5')

# save the label encoder to disk
f = open('NUAA_dataset.pickle', "wb")
f.write(pickle.dumps(le))
f.close()


actual = model.predict(testX)
actual = np.argmax(actual, axis=1) # axis 1 = rows, axis 0 = columns
""" argmax returns the index of the maximum value in each of the rows in the model"""
results = confusion_matrix(np.argmax(testY, axis=1), actual)
report_1 = classification_report(np.argmax(testY, axis=1), actual, target_names=['test data' , 'actual'])
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
