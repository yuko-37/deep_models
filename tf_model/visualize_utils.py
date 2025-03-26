import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(train_acc, test_acc):
    plt.plot(np.squeeze(train_acc))
    plt.xlabel('Epoch per 10')
    plt.ylabel('Accuracy')
    plt.plot(np.squeeze(test_acc))
    plt.show()


def plot_costs(costs):
    plt.plot(np.squeeze(costs))
    plt.xlabel('epochs (per 10)')
    plt.ylabel('Cost')
    plt.title('Learinig rate = 0.0001')
    plt.show()


def plot_images(images, labels):
    img_iter = iter(images)
    label_iter = iter(labels)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(next(img_iter).numpy().astype('uint8'))
        plt.title(next(label_iter).numpy().astype('uint8'))
        plt.axis("off")
    plt.show()
