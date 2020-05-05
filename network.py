import keras

def build(width, height, depth, classes):

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
	
	return model
