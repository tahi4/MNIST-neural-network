import tensorflow as tf


# PREPROCESSING 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #returns two tuples w loading and testing data

#NORMALISING - scale down 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# NEURAL NETWORK

model = tf.keras.models.Sequential() 

#LAYERS

model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #INPUT #makes grid a straight line #input
model.add(tf.keras.layers.Dense(128, activation='relu')) #HIDDEN #relu = rectify linear unit 
model.add(tf.keras.layers.Dense(128, activation='relu')) #HIDDEN
model.add(tf.keras.layers.Dense(10, activation='softmax')) #OUTPUT #represent each digit 0-9 #softmax probability of it being a digit is from 0-1


# training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3) #epochs = training cycles #increases accuracy

model.save('handwritten.model') #saved model

