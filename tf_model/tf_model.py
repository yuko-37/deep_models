import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from data_utils import load_datasets
from visualize_utils import *


def initialize_parameters(layars_dims=(12288, 25, 12, 6)):
    parameters = {}
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    L = len(layars_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(initializer(shape=[layars_dims[l], layars_dims[l - 1]]))
        parameters['b' + str(l)] = tf.Variable(initializer(shape=[layars_dims[l], 1]))

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = W1 @ X + b1
    A1 = tf.keras.activations.relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = tf.keras.activations.relu(Z2)
    Z3 = W3 @ A2 + b3

    return Z3


def compute_loss(logits, labels):
    logits_T = tf.transpose(logits)
    labels_T = tf.transpose(labels)

    if logits_T.shape[1] != 6 or labels_T.shape[1] != 6:
        print(f"logits_T.shape = {logits_T.shape}")
        print(f"labels_T.shape = {labels_T.shape}")

        raise Exception("Bad shapes. 6 is expected as shape[0]")

    loss = tf.keras.losses.categorical_crossentropy(labels_T, logits_T, from_logits=True)
    loss = tf.reduce_sum(loss)
    return loss


def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1, ])
    return image


def one_hot_matrix(label, C=6):
    one_hot = tf.one_hot(label, C, axis=-1)
    one_hot = tf.reshape(one_hot, [-1, ])
    return one_hot


def prepare_data():
    train_x, train_y, test_x, test_y = load_datasets()

    x_train = tf.data.Dataset.from_tensor_slices(train_x)
    y_train = tf.data.Dataset.from_tensor_slices(train_y)

    x_test = tf.data.Dataset.from_tensor_slices(test_x)
    y_test = tf.data.Dataset.from_tensor_slices(test_y)

    X = x_train.map(normalize)
    Y = y_train.map(one_hot_matrix)

    X_test = x_test.map(normalize)
    Y_test = y_test.map(one_hot_matrix)

    # print(f"type X_test = {type(X_test)}, cardinality = {X_test.cardinality()}")
    # print(f"type Y_test = {type(Y_test)}, cardinality = {Y_test.cardinality()}")


    # X1 = next(iter(X))
    # print(X1)
    # Y1 = next(iter(Y))
    # print(Y1)

    # X1_test = next(iter(X_test))
    # print(X1_test)
    # Y1_test = next(iter(Y_test))
    # print(Y1_test)

    return X, Y, X_test, Y_test


def run_model(mini_batch_size=32, learning_rate=0.0001, num_epochs=100, printCost=True):
    new_train_x, new_train_y, new_test_x, new_test_y = prepare_data()

    costs = []
    train_acc = []
    test_acc = []

    parameters = initialize_parameters()

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    dataset = tf.data.Dataset.zip((new_train_x, new_train_y))
    test_dataset = tf.data.Dataset.zip((new_test_x, new_test_y))

    m = dataset.cardinality().numpy()
    if m != 1080:
        print(f"m = {m}")
        raise Exception("Bad m value")

    minibatches = dataset.batch(mini_batch_size).prefetch(8)
    test_minibatches = test_dataset.batch(test_dataset.cardinality().numpy()).prefetch(8)
    print(f"minibatches cardinality = {minibatches.cardinality()}")
    print(f"test_minibatches cardinality = {test_minibatches.cardinality()}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        train_accuracy.reset_state()

        for (mini_X, mini_Y) in minibatches:

            with tf.GradientTape() as tape:
                Z3 = forward_propagation(tf.transpose(mini_X), parameters)
                mini_batch_loss = compute_loss(Z3, tf.transpose(mini_Y))

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(mini_batch_loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_loss += mini_batch_loss

            train_accuracy.update_state(mini_Y, tf.transpose(Z3))

        epoch_loss /= m

        if (epoch % 10 == 0 or epoch == num_epochs-1) and printCost:
            print(f"Cost after {epoch} epoch: {epoch_loss}")
            costs.append(epoch_loss)
            # expected cost 0: 1.830244
            # expected cost 10: 1.552390
            # expected cost 50: 0.946186
            # expected cost 90: 0.744699
            print(f"Train accuracy: {train_accuracy.result()}")

            for (mini_test_X, mini_test_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(mini_test_X), parameters)
                test_accuracy.update_state(mini_test_Y, tf.transpose(Z3))

            print(f"Test accuracy: {test_accuracy.result()}")
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_state()
    # Cost after epoch 0: 1.830244
    # Train accuracy: tf.Tensor(0.17037037, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.2, shape=(), dtype=float32)
    # Cost after epoch 10: 1.552390
    # Train accuracy: tf.Tensor(0.35925925, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.30833334, shape=(), dtype=float32)
    # Cost after epoch 20: 1.347577
    # Train accuracy: tf.Tensor(0.5083333, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.45, shape=(), dtype=float32)
    # Cost after epoch 30: 1.162699
    # Train accuracy: tf.Tensor(0.6111111, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.51666665, shape=(), dtype=float32)
    # Cost after epoch 40: 1.035301
    # Train accuracy: tf.Tensor(0.6574074, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.55, shape=(), dtype=float32)
    # Cost after epoch 50: 0.946186
    # Train accuracy: tf.Tensor(0.6787037, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.6166667, shape=(), dtype=float32)
    # ...
    # Cost after epoch 90: 0.744699
    # Train accuracy: tf.Tensor(0.7546296, shape=(), dtype=float32)
    # Test_accuracy: tf.Tensor(0.69166666, shape=(), dtype=float32)
    return parameters, costs, train_acc, test_acc


if __name__ == '__main__':
    # investigate_data()
    parameters, costs, train_acc, test_acc = run_model(num_epochs=100)

    plot_costs(costs)
    plot_accuracy(train_acc, test_acc)

    # prepare_data()