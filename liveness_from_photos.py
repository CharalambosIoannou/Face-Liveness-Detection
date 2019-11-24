import pickle
import time

import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import glob

"""
Domain shifting problem- using a different dataset
look for domain adaptation algorithm


"""


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading no_glasses_model")
model = load_model('NUAA_dataset.h5')
le = pickle.loads(open('NUAA_dataset.pickle', "rb").read())

labels1 = ["ImposterRaw", "ClientRaw"]
expected = []
actual = []
for label_name in labels1:
	print('Doing label: ' , label_name)
	for imagePath in glob.iglob(f'test_dataset/raw/{label_name}/*/*.jpg'):
		print(imagePath)
		if (label_name == 'ImposterRaw'):
			expected.append(0)
		else:
			expected.append(1)
		frame = cv2.imread(imagePath)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		faces = faceCascade.detectMultiScale(
				gray,
				scaleFactor=1.3,
				minNeighbors=3,
				minSize=(30, 30)
		)
		for (x, y, w, h) in faces :
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#get pixel locations of the box to extract face
			roi_color = frame[y :y + h, x :x + w]
			face = frame[y :y + h, x :x + w]
			# cv2.imshow("Frame1", face)
			face = cv2.resize(face, (32, 32))
			# cv2.imshow("Frame2", face)
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)
		
		
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
			actual.append(label)


print("expected: " , expected)
print("actual: " , actual)
counter = 0
for i in range(0,len(actual)):
	if (expected[i] != actual[i]):
		counter = counter +1
counter = counter + (len(expected) - len(actual))
print("diference: " , counter)

print("len expected: " , len(expected))
print("len actual: " , len(actual))

accuracy = (counter / len(actual)) * 100
print("Accuracy: " , accuracy)
