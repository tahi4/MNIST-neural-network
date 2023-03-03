import tensorflow as tf


# testing prep
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #returns two tuples w loading and testing data

x_test = tf.keras.utils.normalize(x_test, axis=1)

# testing model
model = tf.keras.models.load_model('handwritten.model')
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

