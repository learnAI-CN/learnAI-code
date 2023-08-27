# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义sigmoid函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)


# 构建神经网络
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    # 前向传播
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = sigmoid(self.z2)
        return self.a2

    # 反向传播
    def backward(self, X, y, output):
        self.error = y - output
        self.delta2 = self.error * sigmoid_derivative(output)
        self.error1 = self.delta2.dot(self.weights2.T)
        self.delta1 = self.error1 * sigmoid_derivative(self.a1)

        # 更新权重和偏置
        self.weights2 += self.a1.T.dot(self.delta2)
        self.bias2 += np.sum(self.delta2, axis=0, keepdims=True)
        self.weights1 += X.T.dot(self.delta1)
        self.bias1 += np.sum(self.delta1, axis=0)

    # 训练模型
    def train(self, X, y, epochs):
        loss_list = []
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean(np.square(y - output))
            loss_list.append(loss)
        plt.plot(loss_list)
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    # 预测
    def predict(self, X):
        return self.forward(X).round()


# 训练数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 创建神经网络
nn = NeuralNetwork(2, 3, 1)

# 训练模型
nn.train(X, y, 10000)

# 预测
print(nn.predict(X))