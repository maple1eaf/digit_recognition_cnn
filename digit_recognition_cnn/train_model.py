import cv2
import tensorflow as tf

from digit_recognition_cnn.config import MODEL_NAME


def show_image(image):
    cv2.namedWindow("Image")
    cv2.moveWindow("Image", 40, 30)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

# show_image(x_train[0])

model = tf.keras.models.Sequential()

model.add(
    tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
    )
)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(strides=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x=x_train, y=y_train, batch_size=64, epochs=10)

model.save(filepath=MODEL_NAME)


# print(x_train.shape)
# print(x_train[0][14])
# print(y_train.shape)
# print(y_train[0])
