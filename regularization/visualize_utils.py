import matplotlib.pyplot as plt


def scatter(x, y):
    plt.scatter(x[0, :], x[1, :], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations per thousand')
    plt.title('Cost Function')
    plt.show()