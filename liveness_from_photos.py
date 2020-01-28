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


print("[INFO] loading images...")
actual = []
expected = []
labels = []
labels1 = ["fake", "real"]
for label_name in labels1:
	print('Doing label: ' , label_name)
	for imagePath in glob.iglob(f'test_dataset/my_face/{label_name}/*/*.jpg'):
		print(imagePath)
		if (label_name == 'fake'):
			expected.append(0)
		else:
			expected.append(1)
		# extract the class label from the filename, load the image and
		# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
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
			# face = cv2.resize(face, (32, 32))
			# cv2.imshow("Frame2", face)
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
			print("label: ", label)
			actual.append(label)
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (x, y - 10),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	
			cv2.rectangle(frame, (x, y), (x + w, y + h),
		              (0, 0, 0), 2)

		# show the output frame and wait for a key press
		cv2.imshow("Frame", frame)
		cv2.waitKey(0)


print("expected: " , expected)
print("actual: " , actual)
# counter = 0
# for i in range(0,len(actual)):
# 	if (expected[i] != actual[i]):
# 		counter = counter +1
# counter = counter + (len(expected) - len(actual))
# print("diference: " , counter)
#
# print("len expected: " , len(expected))
# print("len actual: " , len(actual))
#
# accuracy = (counter / len(actual)) * 100
# print("Accuracy: " , accuracy)
