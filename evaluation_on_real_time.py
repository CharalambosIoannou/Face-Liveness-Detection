import pickle
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef,classification_report
from collections import Counter

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading model")
model = load_model('with_chris1_85_straight.h5' ,custom_objects={"tf": tf}) #, custom_objects={"tf": tf}
le = pickle.loads(open('with_chris1_85_straight.pickle', "rb").read())


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
		print("preds: ", preds)
		print("j: " , j)
		label = [0,1][j]

		print(int(time.time()) - int(start))
		sub = int(time.time()) - int(start)
		if ( sub <=15):
			print("Adding")
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
	if key == ord("q") :
		break

print(len(pred))

if choice == "real":
	actual = [1 for i in range(len(pred))]
else:
	actual = [0 for i in range(len(pred))]
print(pred)
print(actual)

y_pred1= Counter(pred)
y_test1 = Counter(actual)
most_common_y_pred = y_pred1.most_common(1)
most_common_y_test = y_test1.most_common(1)
majority_voting = most_common_y_pred[0][0]
if (majority_voting == 1):
	majority_voting = 'real'
else:
	majority_voting = 'fake'




y_pred = np.array(pred)
y_test = np.array(actual)
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

print(len(y_test))
print(y_pred.mean())
print("acc: " ,acc)
print("Majority Voting: " , majority_voting)


# do a bit of cleanup
cv2.destroyAllWindows()

