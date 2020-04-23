import keras
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import LSTM,ConvLSTM2D, TimeDistributed


def build(width, height, depth, classes):
	# initialize the model along with the input shape to be
	# "channels last" and the channels dimension itself
	model = keras.models.Sequential()
	model.add(keras.layers.convolutional.Conv2D(16, (3, 3), padding="same",
		input_shape=(height, width, depth)))
	model.add(keras.layers.LeakyReLU(alpha=0.3))
	model.add(keras.layers.normalization.BatchNormalization(axis=-1))
	model.add(keras.layers.convolutional.Conv2D(16, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.3))
	model.add(keras.layers.normalization.BatchNormalization(axis=-1))
	model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.core.Dropout(0.25))

	model.add(keras.layers.convolutional.Conv2D(32, (3, 3), padding="same")) # 32 is feature maps. Memorizing different patters
	model.add(keras.layers.LeakyReLU(alpha=0.3))
	model.add(keras.layers.normalization.BatchNormalization(axis=-1))
	model.add(keras.layers.convolutional.Conv2D(32, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.3))
	model.add(keras.layers.normalization.BatchNormalization(axis=-1))
	model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.core.Dropout(0.25))

	# downsample image, leave same kernel size
	model.add(keras.layers.convolutional.Conv2D(64, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.3))
	model.add(keras.layers.normalization.BatchNormalization(axis=-1))
	model.add(keras.layers.convolutional.Conv2D(64, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.3))
	model.add(keras.layers.normalization.BatchNormalization(axis=-1))
	model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.core.Dropout(0.25))


	model.add(keras.layers.core.Flatten())
	model.add(keras.layers.core.Dense(128))
	
	# import keras
	# model = keras.models.Sequential()
	#
	# model.add(
	#     TimeDistributed(
	#         keras.layers.convolutional.Conv2D(64, (3, 3), activation='relu'),
	#         input_shape=(10, width, height, 1)
	#     )
	# )
	# model.add(TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(1, 1))))
	#
	# model.add(TimeDistributed(keras.layers.convolutional.Conv2D(128, (4,4), activation='relu')))
	# model.add(TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))
	#
	# model.add(TimeDistributed(keras.layers.convolutional.Conv2D(256, (4,4), activation='relu')))
	# model.add(TimeDistributed(keras.layers.convolutional.MaxPooling2D((2, 2), strides=(2, 2))))
	#
	# # extract features and dropout
	# model.add(TimeDistributed(keras.layers.core.Flatten()))
	# model.add(keras.layers.core.Dropout(0.5))
	#
	# # input to LSTM
	# model.add(LSTM(256, return_sequences=False, dropout=0.5))
	#
	# # classifier with sigmoid activation for multilabel
	# model.add(keras.layers.core.Dense(2, activation='sigmoid'))
	
	# return the constructed network architecture
	return model
