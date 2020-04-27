import pickle
import time

import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import glob
from collections import Counter


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = load_model('model_save.h5')
le = pickle.loads(open('model_save.pickle', "rb").read())
f = []
def multiple_detection():
	a= ['test_detectedface','test_face_both_eyes','test_face_no_left_eye','test_face_no_right_eye','test_face_no_mouth','test_face_no_nose']
	
	for dataset in a:
		print("[INFO] loading images...")
		actual = []
		expected = []
		labels = []
		real_fake = []
		dict = {}
		labels_t = ["ImposterFace", "ClientFace"]
		for label_name in labels_t:
			print('Doing label: ' , label_name)
			for imagePath in glob.iglob(f'dataset/{dataset}/{label_name}/*/*.jpg'):
					print(imagePath)
					image = cv2.imread(imagePath)
					frame = imutils.resize(image, width=600)
					frame = cv2.resize(frame, (32, 32))
					frame = frame.astype("float") / 255.0
					frame = img_to_array(frame)
					frame = np.expand_dims(frame, axis=0)
					preds = model.predict(frame)[0]
					j = np.argmax(preds)
					label = [0,1][j]
					if (label_name == "ImposterFace"):
						label1 = 0
					else:
						label1= 1
					actual.append(label1)
					expected.append(label)
					dict[imagePath] = label
					real_fake.append(label1)
				
		counter = 0
		for i in range (0,len(expected)):
			if (expected[i] == actual[i]):
				counter = counter +1
		
		
		acc = round((counter / len(expected)) * 100,2)
		f.append((dataset,acc))
	print(f)
	return f


def single_image(image_inp):
	image = cv2.imread(image_inp)
	image = imutils.resize(image, width=600)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.3,
			minNeighbors=3,
			minSize=(30, 30)
	)
	if len(faces) == 0:
		print("No face in front of the camera")
		return
	for (x, y, w, h) in faces :
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		#get pixel locations of the box to extract face
		face = image[y :y + h, x :x + w]
		face = cv2.resize(face, (32, 32))
		face =img_to_array( face.astype("float") / 255.0)
		face = np.expand_dims(face, axis=0)
		preds = model.predict(face)
		preds = preds[0]
		j = np.argmax(preds)
		label = [0,1][j]
		if (label == 1):
			label='real'
		else:
			label = 'fake'
		perc = round(preds[j] *100,2)
		label = f"{label}- conf: {perc}%"
		cv2.putText(image, label, (x, y - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

		cv2.rectangle(image, (x, y), (x + w, y + h),
		              (0, 0, 0), 2)
	
	print(label)
	# image = cv2.resize(image, (1000,700))
	cv2.imshow("img: " , image)
	cv2.waitKey(0)
	return label

# single_image()
# multiple_detection()

