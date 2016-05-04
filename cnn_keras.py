

# 

from keras.models import Sequential

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.layers.core import Activation



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

model.add( 
	Convolution2D(5, 4, 4, border_mode='same',  input_shape=(img_channels, img_rows, img_cols)) 
	)
# layer 1 activation function
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))