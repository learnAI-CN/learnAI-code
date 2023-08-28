# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import pickle
import gzip

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        使用反向传播算法对单个小批量数据进行梯度下降，更新网络的权重和偏置。
        其中，mini_batch 是一个包含元组 (x, y) 的列表，eta 是学习率。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        返回一个元组 (nabla_b, nabla_w)，表示代价函数 C_x 的梯度。
        nabla_b 和 nabla_w 是按层排列的 numpy 数组列表，
        类似于 self.biases 和 self.weights。
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回神经网络输出正确结果的测试输入数量。
        需要注意的是，假设神经网络的输出是最终层中具有最高激活的神经元的索引。
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def load_data():
    """
    将 MNIST 数据返回为一个元组，包含训练数据、验证数据和测试数据。
    其中，training_data 作为一个元组返回，包含两个条目。
    第一个条目包含实际的训练图像，这是一个包含 50,000 个条目的 numpy ndarray。
    每个条目又是一个包含 784 个值的 numpy ndarray，表示单个 MNIST 图像中的 28 * 28 = 784 个像素。
    training_data 元组的第二个条目是一个包含 50,000 个条目的 numpy ndarray，
    这些条目仅是元组第一个条目中对应图像的数字值（0...9）。
    validation_data 和 test_data 也类似，但每个元组仅包含 10,000 个图像。
    这是一种不错的数据格式，但为了在神经网络中使用更方便，需要对 training_data 的格式进行一些修改，
    这在下面的包装函数 load_data_wrapper() 中完成。
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    返回一个元组，包含 (training_data, validation_data, test_data)。
    该函数基于 load_data 实现，但其输出格式更适合我们实现的神经网络。
    具体而言，training_data 是一个包含 50,000 个二元组 (x, y) 的列表。
    其中，x 是一个包含输入图像的 784 维 numpy.ndarray，y 是一个表示 x 对应正确数字的 10 维 numpy.ndarray 单位向量。
    validation_data 和 test_data 分别是包含 10,000 个二元组 (x, y) 的列表。
    在每个二元组中，x 是一个包含输入图像的 784 维 numpy.ndarray，y 是与 x 对应的分类，即对应于 x 的数字值（整数）。
    显然，这意味着我们在训练数据和验证/测试数据上使用略有不同的格式。这些格式被证明是我们的神经网络代码中最方便使用的。
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    返回一个 10 维的单位向量，其中第 j 个位置为 1.0，其他位置为零。
    这用于将一个数字（0...9）转换为神经网络的相应期望输出。
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    # - read the input data:
    training_data, validation_data, test_data = load_data_wrapper()
    training_data = list(training_data)

    # define Network
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
