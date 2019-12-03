import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)


def save(name):
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def plot(x, y):
    plt.plot(x, y)
    plt.ylim(-1.2, 1.2)
    plt.xlim(-0.2, 2 * np.pi + 0.4)


def scatter(x, y, letter):
    plt.scatter(x, y)
    plt.ylim(-1.2, 1.2)
    plt.xlim(-0.2, 2 * np.pi + 0.4)
    for i in range(len(x)):
        plt.annotate('${}_{{{}}} = ({:.1f}, {:.1f})$'.format(letter, i + 1, x[i], y[i]), (x[i] - 0.1, y[i] - 0.14))


def noise(num, eps=0.04):
    x = np.linspace(eps + 0.01, 2 * np.pi - eps - 0.01, num) + np.random.uniform(low=-eps, high=eps, size=num)
    y = np.sin(x)
    return x, y


true_x = np.linspace(0, 2 * np.pi, 100)
true_y = np.sin(true_x)
plot(true_x, true_y)
save('Figure_1.png')

sample1_x, sample1_y = noise(6)
plot(true_x, true_y)
scatter(sample1_x, sample1_y, 'x')
save('Figure_2.png')

sample2_x, sample2_y = noise(10)
plot(true_x, true_y)
scatter(sample2_x, sample2_y, 'y')
save('Figure_3.png')
