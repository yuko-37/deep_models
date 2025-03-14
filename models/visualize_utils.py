import matplotlib.pyplot as plt
import colorsys

def plot_one(costs, title):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(title)
    plt.show()


def plot_many(costs_set, labels, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    n = len(costs_set)
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)

    for i, costs in enumerate(costs_set):
        plt.plot(costs, color=colors[i], label=labels[i])

    ax.set_ylabel('cost')
    ax.set_xlabel('iterations (per hundreds)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_image(image):
    plt.imshow(image)
    plt.show()

