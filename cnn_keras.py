

# 

from keras.models import Sequential

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.layers.core import Activation, Flatten, Dense, Dropout

from keras.utils import np_utils

from keras.optimizers import SGD

import numpy as np


# data Preporocessing
# training_targets shape (35887, 7)     int   32
# training_inputs  shape (35887, 2304)  float 32      
import pandas as pd 
csv_data = pd.read_csv('./data/fer2013/fer2013.csv')

training_targets = csv_data.ix[:,0].values.astype('int32')
training_targets = np_utils.to_categorical(training_targets) 


training_inputs= []
for value in csv_data.ix[:,1].values :
    value = ( np.array(value.split()) ).astype('float32')

    training_inputs.append(value)

training_inputs = np.array(training_inputs)


training_inputs = training_inputs.reshape(training_inputs.shape[0], 1, 48, 48)





# image properties
img_rows = 48
img_cols = 48
img_channels = 1


#keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)

	# initiate mothership
model = Sequential()



########
#layer 1
########
#border_mode='same',  
model.add( 
	Convolution2D(32, 5, 5, input_shape=(img_channels, img_rows, img_cols)) 
	)
# layer 1 activation function
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(32*6*6, 128, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#final layer   
model.add(Dense(128, 7, init='lecun_uniform'))
model.add(Activation('softmax'))


# train using gradient descent ... correct the weights and biases
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(training_inputs[0:29999], training_targets[0:29999], nb_epoch=10, batch_size=1000, validation_split=0.1, show_accuracy=True)

