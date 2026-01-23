import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D,
                                     BatchNormalization,
                                     AveragePooling2D, MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform


def load_signs_data():
    train_ds = h5py.File('datasets/train_signs.h5')
    train_x = np.array(train_ds['train_set_x'])
    train_y = np.array(train_ds['train_set_y'])

    test_ds = h5py.File('datasets/test_signs.h5')
    test_x = np.array(test_ds['test_set_x'])
    test_y = np.array(test_ds['test_set_y'])

    classes = np.array(train_ds["list_classes"][:])

    return train_x, train_y, test_x, test_y, classes


def identity_block(X, f, filters, initializer=random_uniform):
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)  # Default axis
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=1, padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, initializer=glorot_uniform):
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=1, padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=s, padding='valid', kernel_initializer=initializer(seed=0))(
        X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6, training=False):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = AveragePooling2D(pool_size=2)(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X)

    return model


def train():
    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    # print(model.summary())

    np.random.seed(1)
    tf.random.set_seed(2)
    opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_data()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = to_categorical(Y_train_orig, 6)
    Y_test = to_categorical(Y_test_orig, 6)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    model.fit(X_train, Y_train, epochs = 20, batch_size=32)
    print('Evaluation:')
    model.evaluate(X_test, Y_test)
    print()
    model_name = "res_net_sign_recognizer"
    model.save(f"model_parameters/{model_name}.keras")
    print(f"Model is saved: {model_name}.keras")


def recognize_sign(image_file):
    img_path = "images/" + image_file
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    model = tf.keras.models.load_model("model_parameters/res_net_sign_recognizer.keras")
    prediction = model.predict(x)
    print(prediction)
    print("Class:", np.argmax(prediction))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # train()
    recognize_sign("hand-5.jpg")
