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
model = load_model('NUAA_dataset_final.h5')
le = pickle.loads(open('NUAA_dataset_final.pickle', "rb").read())

# a = ['face_no_left_eye','face_no_right_eye','face_no_mouth','face_no_nose']
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
				label = le.classes_[j]
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

#both eyes accuracy: 56.85%
#no-mouth accuracy : 60.06%
#no-nose accuracy : 65.42%
#no-right eye accuracy : 48.35%
#no-left eye accuracy : 51.69%


""" "
'test_face_both_eyes': 56.8593

'test_face_no_left_eye': 51.6923

'test_face_no_right_eye': 48.3516

'test_face_no_mouth': 57.5385

'test_face_no_nose':  87.9121

"""
