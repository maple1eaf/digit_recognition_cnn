import tensorflow as tf

from digit_recognition_cnn.config import MODEL_NAME

"""
training dataset:
60000 28*28 gray(0-255) image

testing dataset:
10000 28*28 gray(0-255) image
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
# print(x_train[0][14])
x_test = x_test / 255

model = tf.keras.models.load_model(filepath=MODEL_NAME)

loss, accuracy = model.evaluate(x_test, y_test)

print(x_test.shape)
print(f"{loss=}")
print(f"{accuracy=}")
