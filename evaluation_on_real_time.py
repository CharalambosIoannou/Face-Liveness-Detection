import pickle
import time
from collections import Counter

import cv2
import imutils
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array



faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading model")
model = load_model('model_save.h5' ,custom_objects={"tf": tf}) #, custom_objects={"tf": tf}
le = pickle.loads(open('model_save.pickle', "rb").read())


print("Starting camera")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
pred = []
actual = []
choice = input("Fake or real? \n")
start = time.time()
flag = False
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
		key = cv2.waitKey(1) & 0xFF
		if (key == ord(" ")):
			cv2.imwrite('saved_img '+str(int(round(preds[j] * 100)))+'.jpg',face)
		face = cv2.resize(face, (32, 32))

		# cv2.imshow("Frame2", face)
		face =img_to_array( face.astype("float") / 255.0)
		face = np.expand_dims(face, axis=0)
		preds = model.predict(face)
		preds = preds[0]
		j = np.argmax(preds)
		label = [0,1][j]

		sub = int(time.time()) - int(start)
		if ( sub <=15):
			pred.append(label)
		else:
			flag = True
			break
		
		# draw the label and bounding box on the frame
		if (label == 1):
			label='real'
		else:
			label = 'fake'
			
		perc = round(preds[j] *100,2)
		label = f"{label}- conf: {perc}%"
		
		
		cv2.putText(frame, label, (x, y - 10),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

		cv2.rectangle(frame, (x, y), (x + w, y + h),
		              (0, 0, 0), 2)
	if flag:
		break
	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	# if the `q` key was pressed, break from the loop
	if key == ord("q") :
		break


if choice == "real":
	actual = [1 for i in range(len(pred))]
else:
	actual = [0 for i in range(len(pred))]

y_pred1= Counter(pred)
y_test1 = Counter(actual)
most_common_y_pred = y_pred1.most_common(1)
most_common_y_test = y_test1.most_common(1)
majority_voting = most_common_y_pred[0][0]
if (majority_voting == 1):
	majority_voting = 'real'
else:
	majority_voting = 'fake'



counter = 0
for i in range (0,len(pred)):
	if (pred[i] == actual[i]):
		counter = counter +1


acc = round((counter / len(pred)) * 100,2)
y_pred = np.array(pred)
y_test = np.array(actual)

print("Accuracy: " ,acc)
print("Majority Voting: " , majority_voting)

cv2.destroyAllWindows()

