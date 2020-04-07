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

def init(image):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("face_occlusion/shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	
	image = imutils.resize(image, width=500)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# loop over the face detections

	# rect = dlib.rectangle(x, y, x+w, y+h )
	rects = detector(gray, 1)[0]
	shape = predictor(gray, rects)
	shape = face_utils.shape_to_np(shape)
	return image, shape

def occlude_region(img,region):
	face_features = {
	"mouth": (48, 68),
	"right_eyebrow": (17, 22),
	"left_eyebrow": (22, 27),
	"right_eye": (36, 42),
	"left_eye": (42, 48),
	"nose": (27, 36),
	}
	feature =  face_features.get(region)
	image , shape = init(img)
	clone = image.copy()
	(x, y, w, h) = cv2.boundingRect(np.array([shape[feature[0]:feature[1]]]))
	roi = image[y:y + h, x:x + w]
	# roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
	cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 0, 0), -1)
	# cv2.imshow("roi: " , roi)
	return x, y, w, h

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Loading model")
model = load_model('NUAA_dataset_final_85.h5' ,custom_objects={"tf": tf}) #, custom_objects={"tf": tf}
le = pickle.loads(open('NUAA_dataset_final_85.pickle', "rb").read())


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
print("Majority Voting: " , majority_voting)



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



# do a bit of cleanup
cv2.destroyAllWindows()

