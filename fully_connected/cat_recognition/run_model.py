import numpy as np
import random

from utils.data_utils import load_reshaped_data, load_data
from fully_connected.cat_recognition.numpy_FC_model import L_layers_model, L_model_forward
from utils.visualize_utils import *
from PIL import Image
from utils.save_load_parameters import load_from, save_to


def predict(X, Y, parameters, dataset_name="", print_accuracy=True):
    m = X.shape[1]
    probs, caches = L_model_forward(X, parameters)
    predictions = (probs > 0.5).astype(int)
    accuracy = np.sum((predictions == Y) / m)
    if print_accuracy:
        print(f"Accuracy {dataset_name} = {accuracy:.4}")
    return predictions


def train(layers_dims=(12288, 20, 7, 5, 1), plot_cost=True):
    np.random.seed(1)
    train_x, train_y, test_x, test_y = load_reshaped_data()
    num_iterations = 2000
    learning_rate = 0.0075
    print(f"Start learning {len(layers_dims) - 1}-layers model:\
layers_dims = {layers_dims}; num_iterations = {num_iterations}; learning_rate = {learning_rate}")

    parameters, costs = L_layers_model(train_x, train_y, layers_dims, num_iterations, learning_rate)
    save_to(parameters, layers_dims, file_from_dims(layers_dims))

    print()
    predict(train_x, train_y, parameters, 'train set')
    predict(test_x, test_y, parameters, 'test set')
    print()

    if plot_cost:
        plot_one(costs, label(learning_rate, layers_dims))


def label(rate, dims):
    return f"Learning_rate={rate}, layers_dims={dims}"


def file_from_dims(layers_dims):
    filename = ('parameters/model-' + str(layers_dims).replace('(', '')
                .replace(')', '').replace(', ', '-') + '.pkl')
    return filename


def train_many_models():
    np.random.seed(1)
    train_x, train_y, test_x, test_y = load_reshaped_data()
    costs_set = []
    labels_set = []

    num_iterations = 100

    # one
    layers_dims = (train_x.shape[0], 20, 7, 5, 1)
    learning_rate = 0.0075
    print(
        f"Start learning {len(layers_dims) - 1}-layers model: layers_dims = {layers_dims}; num_iterations = {num_iterations}; learning_rate = {learning_rate}")
    parameters, costs = L_layers_model(train_x, train_y, layers_dims, num_iterations, learning_rate)
    print()
    predict(train_x, train_y, parameters, 'train set')
    predict(test_x, test_y, parameters, 'test set')
    print()
    costs_set.append(costs)
    labels_set.append(label(learning_rate, layers_dims))

    # two
    layers_dims = (train_x.shape[0], 20, 7, 5, 1)
    learning_rate = 0.005
    print(
        f"Start learning {len(layers_dims) - 1}-layers model: layers_dims = {layers_dims}; num_iterations = {num_iterations}; learning_rate = {learning_rate}")
    parameters, costs = L_layers_model(train_x, train_y, layers_dims, num_iterations, learning_rate)
    print()
    predict(train_x, train_y, parameters, 'train set')
    predict(test_x, test_y, parameters, 'test set')
    print()
    costs_set.append(costs)
    labels_set.append(label(learning_rate, layers_dims))

    # three
    layers_dims = (train_x.shape[0], 20, 7, 5, 1)
    learning_rate = 0.001
    print(
        f"Start learning {len(layers_dims) - 1}-layers model: layers_dims = {layers_dims}; num_iterations = {num_iterations}; learning_rate = {learning_rate}")
    parameters, costs = L_layers_model(train_x, train_y, layers_dims, num_iterations, learning_rate)
    print()
    predict(train_x, train_y, parameters, 'train set')
    predict(test_x, test_y, parameters, 'test set')
    print()
    costs_set.append(costs)
    labels_set.append(label(learning_rate, layers_dims))

    plot_many(costs_set, labels_set, "Costs")


def random_train_cat():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    indexes = []
    for i, y in enumerate(np.squeeze(train_set_y_orig)):
        if y == 1:
            indexes.append(i)

    img = train_set_x_orig[random.choice(indexes)]
    return img


def recognize_image_array(img_array, parameters_filename):
    img_array_flatten = img_array.reshape(1, -1).T
    X = img_array_flatten / 255
    print(X.shape)
    assert (X.shape == (12288, 1))

    parameters = load_from(parameters_filename)
    prediction = predict(X, [1], parameters, print_accuracy=False)
    prediction = np.squeeze(prediction)

    print(f"y = {prediction}, your L-layer model predicts a '{'cat' if prediction == 1 else 'non-cat'}' picture.")

    plot_image(img_array)


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


def recognize_image(image_file, parameters_filename):
    image_array = img_array_from_img(image_file)
    recognize_image_array(image_array, parameters_filename)


if __name__ == '__main__':
    layers_dimensions = (12288, 20, 10, 7, 5, 1)
    # train(layers_dimensions)

    recognize_image('ovcharka.jpeg', file_from_dims(layers_dimensions))
    recognize_image('cat-ara.jpg', file_from_dims(layers_dimensions))
    recognize_image('cat-test-2.png', file_from_dims(layers_dimensions))

