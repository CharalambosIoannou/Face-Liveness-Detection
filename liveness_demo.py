import pickle
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import time
from collections import Counter


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading model")
model = load_model('with_chris1_85_straight.h5') #, custom_objects={"tf": tf}
le = pickle.loads(open('with_chris1_85_straight.pickle', "rb").read())


print("Starting camera")
vs = cv2.VideoCapture(0)
time.sleep(2.0)
pred = []
actual = []
start = time.time()
flag = False
found_face = True
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
	if len(faces) == 0:
		found_face = False
		print("No face in front of the camera")
		break
	for (x, y, w, h) in faces :
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		#get pixel locations of the box to extract face
		roi_color = frame[y :y + h, x :x + w]
		face = frame[y :y + h, x :x + w]
		cv2.imshow("Frame1", face)
		key = cv2.waitKey(1) & 0xFF
		if (key == ord(" ")):
			cv2.imwrite('saved_img '+str(int(round(preds[j] * 100)))+'.jpg',face)
		face = cv2.resize(face, (32, 32))

		cv2.imshow("Frame2", face)
		face =img_to_array( face.astype("float") / 255.0)
		face = np.expand_dims(face, axis=0)
		preds = model.predict(face)
		preds = preds[0]
		j = np.argmax(preds)
		label = [0,1][j]
		sub = int(time.time()) - int(start)
		if ( sub <=5):
			pred.append(label)
		else:
			flag = True
			break
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

	if flag:
		break
	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)



	# if the `q` key was pressed, break from the loop
	try:
		if key == ord("q") :
			break
	except:
		print("No face detected")
		found_face = False
		break

if (found_face):
	print(len(pred))
	print(pred)
	y_pred1= Counter(pred)
	most_common_y_pred = y_pred1.most_common(1)
	majority_voting = most_common_y_pred[0][0]
	if (majority_voting == 1):
		majority_voting = 'real'
		cv2.destroyAllWindows()
		print("Majority Voting: " , majority_voting)
		img = cv2.imread('real.jpg')
		cv2.putText(img, 'Real Face', (20, 30),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow("res: " , img)
		cv2.waitKey(0)
	else:
		majority_voting = 'fake'
		cv2.destroyAllWindows()
		print("Majority Voting: " , majority_voting)
		img = cv2.imread('fake.jpg')
		cv2.putText(img, 'Fake Face', (20, 25),
			            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
		cv2.imshow("res: " , img)
		cv2.waitKey(0)



