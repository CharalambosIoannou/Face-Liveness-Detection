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
model = load_model('with_chris1_85_straight.h5')
le = pickle.loads(open('with_chris1_85_straight.pickle', "rb").read())

def multiple_detection():
	a = ['face_no_left_eye','face_no_right_eye','face_no_mouth','face_no_nose']
	a= ['test_detectedface','test_face_both_eyes','test_face_no_left_eye','test_face_no_right_eye','test_face_no_mouth','test_face_no_nose']
	b = {}
	
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
					if (label == 1):
						label='real'
					else:
						label = 'fake'
					if (label_name == "ImposterFace"):
						label1 = 'fake'
					else:
						label1='real'
					dict[imagePath] = label
					real_fake.append(label1)
	
		res = {}
		import pandas as pd
		df = pd.DataFrame(list(dict.items()), columns=['Filename', 'Prediction'])
		df.insert(2, "Actual", real_fake, True)
		df['same'] = np.where((df['Prediction'] == df['Actual']),"y","n")
		acc = df[(df.same == 'y')].count() /  (df[(df.same == 'y')].count() + df[(df.same == 'n')].count())
		print(df)
		print(acc)
		b[dataset] = acc
	print(b)
	return b


def single_image():
	image = cv2.imread("img2.jpg")
	frame = imutils.resize(image, width=600)
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
		roi_color = frame[y :y + h, x :x + w]
		face = frame[y :y + h, x :x + w]
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
		label = "{}: {:.4f}".format(label, preds[j])
		cv2.putText(image, label, (x, y - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

		cv2.rectangle(image, (x, y), (x + w, y + h),
		              (0, 0, 0), 2)
	
	print(label)
	cv2.imshow("img: " , image)
	cv2.waitKey(0)
	return label

single_image()
# multiple_detection()
