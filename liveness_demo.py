import pickle
import time

import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

"""
Domain shifting problem- using a different dataset
look for domain adaptation algorithm


"""

# wears_glasses = input("Do you wear glasses? Type y or n \n")

#TODO register user
# Preliminary definitions (Let X in the training) Experiment part
# Face occlusion detection


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading model")
model = load_model('1_NUAA_dataset.h5')
le = pickle.loads(open('1_NUAA_dataset.pickle', "rb").read())


print("Starting camera")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
while True :

	ret, frame = vs.read()
	frame = imutils.resize(frame, width=600)
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
		cv2.imshow("Frame1", face)
		face = cv2.resize(face, (32, 32))
		cv2.imshow("Frame2", face)
		face = face.astype("float") / 255.0
		face = img_to_array(face)
		face = np.expand_dims(face, axis=0)


		preds = model.predict(face)[0]
		j = np.argmax(preds)
		label = le.classes_[j]

		# draw the label and bounding box on the frame
		if (label == 1):
			label='real'
		else:
			label = 'fake'
		label = "{}: {:.4f}".format(label, preds[j])
		cv2.putText(frame, label, (x, y - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

		cv2.rectangle(frame, (x, y), (x + w, y + h),
		              (0, 0, 0), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q") :
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
