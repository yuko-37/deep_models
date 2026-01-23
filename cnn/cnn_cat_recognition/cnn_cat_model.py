import tensorflow as tf
import tensorflow.keras.layers as tfl
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sequential_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tfl.ZeroPadding2D(padding=1),
        tfl.Conv2D(filters=32, kernel_size=7, strides=1, padding='same'),
        tfl.BatchNormalization(axis=3),
        tfl.ReLU(),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dense(units=1, activation='sigmoid')
    ])

    return model


def functional_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)
    Z1 = tfl.Conv2D(filters=8, kernel_size=3, strides=1, padding="same")(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=2, strides=2, padding="same")(A1)

    Z2 = tfl.Conv2D(filters=16, kernel_size=3, strides=1, padding="same")(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=2, strides=2, padding="same")(A2)

    Z3 = tfl.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(P2)
    A3 = tfl.ReLU()(Z3)
    P3 = tfl.MaxPool2D(pool_size=2, strides=2, padding="same")(A3)

    Z4 = tfl.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(P3)
    A4 = tfl.ReLU()(Z4)
    P4 = tfl.MaxPool2D(pool_size=2, strides=2, padding="same")(A4)

    F = tfl.Flatten()(P4)
    outputs = tfl.Dense(units=1, activation='sigmoid')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


def train_model():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_data()
    X_train = X_train_orig / 255.0
    X_test = X_test_orig / 255.0
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    print()

    input_shape = X_train[0].shape
    model = functional_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, epochs=3, batch_size=16)
    print()
    print('Evaluation:')
    model.evaluate(X_test, Y_test)
    print()
    model_name = "cnn_seq_cat_recognizer"
    model.save(f"model_parameters/{model_name}.keras")
    print(f"Model is saved: {model_name}.keras")


def recognize_image(image_file):
    image_array = img_array_from_img(image_file)
    recognize_image_array(image_array, image_file)


def recognize_images(files):
    shape = (len(files), 64, 64, 3)
    X = np.zeros(shape)
    for i in range(len(files)):
        X[i] = img_array_from_img(files[i])

    recognize_image_arrays(X, files)


def recognize_image_arrays(img_arrays, names):
    X = img_arrays / 255.0

    model = tf.keras.models.load_model("model_parameters/cnn_seq_cat_recognizer.keras")
    predictions = model.predict(X)
    predictions = np.squeeze(predictions)

    for i in range(len(predictions)):
        print(f"y = {predictions[i]}, model predicts {names[i]} as a '{'cat' if predictions[i] > 0.5 else 'non-cat'}' picture.")
    # plt.imshow(img_array)
    # plt.show()


def recognize_image_array(img_array, name):
    X = img_array / 255.0
    X = np.expand_dims(X, axis=0)

    model = tf.keras.models.load_model("model_parameters/cnn_seq_cat_recognizer.keras")
    prediction = model.predict(X)
    prediction = np.squeeze(prediction)

    print(f"y = {prediction}, model predicts {name} as a '{'cat' if prediction > 0.5 else 'non-cat'}' picture.")
    # plt.imshow(img_array)
    # plt.show()


def img_array_from_img(image_file):
    filename = "images/" + image_file
    img = Image.open(filename)
    if filename.endswith(('.jpg', '.jpeg')):
        img = img.resize((64, 64))
    elif filename.endswith('.png'):
        img = img.resize((64, 48))
    else:
        raise Exception(f"Not tested format {image_file}")

    return np.array(img)


if __name__ == '__main__':
    # train_model()
    recognize_images([
        "cat-0.jpg",
        "cat-1.jpg",
        "cat-2.jpg",
        "cat-3.jpg",
        "cat-ara.jpg",
        "cat-internet-1.jpg",
        "cat-manul.jpg",
        "maltipu.jpg",
        "ovcharka.jpeg"])
