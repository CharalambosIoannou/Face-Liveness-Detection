import pickle

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils
import dlib
from imutils import face_utils
import random

model = load_model('NUAA_dataset_final.h5')
le = pickle.loads(open('NUAA_dataset_final.pickle', "rb").read())
def get_accuracy(img):
	frame = cv2.resize(img, (32, 32))
	frame = frame.astype("float") / 255.0
	frame = img_to_array(frame)
	frame = np.expand_dims(frame, axis=0)
	preds = model.predict(frame)[0]
	j = np.argmax(preds)
	label = le.classes_[j]
	return  label,preds[j]



def occlude_convolution_random(final_img,size_box):
	h,w,_ = final_img.shape
	row = random.randint(0,h)
	col = random.randint(0,w)
	final_img = cv2.rectangle(final_img, (row, col), (row+size_box, col+size_box), (255, 255, 255), -1)
	label, perc = get_accuracy(final_img)
	perc =  str(int(round(perc * 100)))
	print("new perc acc: " , perc)
	print("new label: " ,label)
	print("*****************")
	if (label == 1):
		cv2.putText(final_img, perc, (row,col), cv2.FONT_HERSHEY_SIMPLEX ,
           (450 * 450) / (1000 * 1000) , (0, 0, 0) , 1, cv2.LINE_AA)
	else:
		final_img = cv2.rectangle(final_img, (row, col), (row+size_box, col+size_box), (0, 0, 255), -1)
		cv2.putText(final_img, perc, (row,col), cv2.FONT_HERSHEY_SIMPLEX ,
           (450 * 450) / (1000 * 1000) , (0, 0, 0) , 1, cv2.LINE_AA)
	cv2.imshow("Final: " , final_img)
	cv2.waitKey(0)


def occlude_convolution(final_img,size_box):
	c = 14
	l = []
	for i in range (width):
		if (i % (size_box+1) == 0):
			for j in range (height):
				if (j % (size_box+1) == 0):
					
					cv2.imshow("org: " , img)
					img_c = img.copy()
					cv2.imshow("cop: " , img_c)
					occlude =  cv2.rectangle(img_c, (i, j), (i+size_box, j+size_box), (255, 255, 255), -1)
					final_img = cv2.rectangle(final_img, (i, j), (i+size_box, j+size_box), (255, 255, 255), -1)
					cv2.imshow("occ: " , occlude)
					cv2.waitKey(0)
					label, perc = get_accuracy(occlude)
					perc =  str(int(round(perc * 100)))
					print("new perc acc: " , perc)
					print("new label: " ,label)
					print("*****************")
					l.append(int(perc))
					if (label == 1):
						cv2.putText(final_img, perc, (i,j), cv2.FONT_HERSHEY_SIMPLEX ,
		                   (450 * 450) / (1000 * 1000) , (0, 0, 0) , 1, cv2.LINE_AA)
					else:
						final_img = cv2.rectangle(final_img, (i, j), (i+size_box, j+size_box), (0, 0, 255), -1)
						cv2.putText(final_img, perc, (i,j), cv2.FONT_HERSHEY_SIMPLEX ,
		                   (450 * 450) / (1000 * 1000) , (0, 0, 0) , 1, cv2.LINE_AA)
					c = c + 14
					cv2.imshow("Final: " , final_img)
					cv2.waitKey(0)
					

 
def occlude_specific_part(img,region):
	output = occlude_region(img,region)
	label, perc = get_accuracy(output)
	perc =  str(int(round(perc * 100)))
	print("new perc acc: " , perc)
	print("new label: " ,label)
	print("*****************")
	cv2.imshow("afs: " , output)
	cv2.waitKey(0)


	


def init(image):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("face_occlusion/shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	# image = imutils.resize(image, width=500)

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
	cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 255, 255), -1)
	# cv2.imshow("roi: " , roi)
	return clone


image = 'saved_img 99.jpg'

# load img
img = cv2.imread(image)
img_c = img.copy()
width, height,_ = img.shape


label, perc = get_accuracy(img)
perc =  str(int(round(perc * 100)))
print("org perc acc: " , perc)
print("org label: " ,label)

size_box = 30

final_img = img.copy()

occlude_convolution(final_img,size_box)
# occlude_convolution_random(final_img,size_box)
# occlude_specific_ part(final_img,"left_eye")
