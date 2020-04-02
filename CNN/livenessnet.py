# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import LSTM,ConvLSTM2D, TimeDistributed
from keras.layers import LeakyReLU

class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(16, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(LeakyReLU(alpha=0.3))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(LeakyReLU(alpha=0.3))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same")) # 32 is feature maps. Memorizing different patters
		model.add(LeakyReLU(alpha=0.3))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(LeakyReLU(alpha=0.3))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# downsample image, leave same kernel size
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(LeakyReLU(alpha=0.3))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(LeakyReLU(alpha=0.3))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(128))
		
		# import keras
		# model = Sequential()
		#
		# model.add(
		#     TimeDistributed(
		#         Conv2D(64, (3, 3), activation='relu'),
		#         input_shape=(10, width, height, 1)
		#     )
		# )
		# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
		#
		# model.add(TimeDistributed(Conv2D(128, (4,4), activation='relu')))
		# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
		#
		# model.add(TimeDistributed(Conv2D(256, (4,4), activation='relu')))
		# model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
		#
		# # extract features and dropout
		# model.add(TimeDistributed(Flatten()))
		# model.add(Dropout(0.5))
		#
		# # input to LSTM
		# model.add(LSTM(256, return_sequences=False, dropout=0.5))
		#
		# # classifier with sigmoid activation for multilabel
		# model.add(Dense(2, activation='sigmoid'))
		
		# return the constructed network architecture
		return model
