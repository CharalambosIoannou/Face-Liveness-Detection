
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import glob

# code modified from https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/


def init(image):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
	return clone




# image = cv2.imread("../../dataset/raw/ClientRaw/0001/0001_00_00_01_0.jpg")

import os
labels1 = ["ImposterFace", "ClientFace"]
os.mkdir('../dataset/face_no_mouth')
for label_name in labels1:
	print('Doing label: ' , label_name)
	os.mkdir('../dataset/face_no_mouth/'+label_name)
	sub_dirs_imp = [x[0][-4:] for x in os.walk("../dataset/Detectedface/ImposterFace")][1:]
	sub_dirs_client = [x[0][-4:] for x in os.walk("../dataset/Detectedface/ClientFace")][1:]
	for i in sub_dirs_imp:
		print("here")
		print(i)
		try:
			os.mkdir('../dataset/face_no_mouth/'+label_name+ "/" + i)
		except FileExistsError:
			continue
			
	for i in sub_dirs_client:
		print("here1")
		try:
			os.mkdir('../dataset/face_no_mouth/'+label_name+ "/" + i)
		except FileExistsError:
			continue
	print("finished with dirs")
	for imagePath in glob.iglob(f'../dataset/Detectedface/{label_name}/*/*.jpg'):
		print(imagePath)
		# extract the class label from the filename, load the image and
		# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
		image = cv2.imread(imagePath)
		try:
			output = occlude_region(image,"mouth") #133 not recongised
		except IndexError:
			continue
		# cv2.imshow("Image", output)
		cv2.imwrite('../dataset/face_no_mouth/'+label_name + "/"+ imagePath[imagePath.find('00'): imagePath.find('00') + 4] + "/" +os.path.basename(imagePath),output)



