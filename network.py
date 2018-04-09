# -*- coding: utf-8 -*-
"""
network.py
author: Michael Nielsen
note by herb
date:2018-3-31
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import pickle

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        print('network',sizes)
        """ 初始化神经网络，sizes输入格式为[第一层神经元个数，第二层。。。]
           biase为每个神经元的偏移量，weight是神经元的权重数组  """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """ 计算神经网络的输出值，np.dot表示矩阵点乘 """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        print('SGD:',epochs,mini_batch_size,eta)
        """ 使用堆积梯度下降法训练神经网络的主要方法，
            训练数据集的格式是（x，y）其中x是输入，y是训练数据的标签 """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs): #训练次数
            random.shuffle(training_data) #对训练数据集随机排序
            mini_batches = [ #以最小样本作为步长，从训练集切割数据到mini_batches
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: #针对切割后的所有数据进行循环，mini_batch为最小样本集
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                i = self.evaluate(test_data)
                print( "Epoch {0}: {1} / {2} = {3}".format(
                    j, i, n_test, round(i/n_test,6)) )
            else:
                print( "Epoch {0} complete".format(j) )
        self.wb_save() #记录权重和基值

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #按照偏置矩阵大小生成一个为0的矩阵
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: #循环全部最小样本集后，再进行权重和偏置值的调整 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #返回一个给定形状和类型的用0填充的数组 shape:形状
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # 计算每一个节点的值
            zs.append(z)
            activation = sigmoid(z) # 加入激活函数
            activations.append(activation)

        ''' backward pass 最后一层的 w和b 
            因为最后一层没有权重weights因此单独计算，没有放在循环中
        '''
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) #argmax 返回最大数的索引
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate_print(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) #argmax 返回最大数的索引
                        for (x, y) in test_data]
        #统计预测错误的数据特征
        error = []
        for i, (x, y) in enumerate(test_results):
            if (x!=y):
                error.append(test_data[i][0])
        error = np.insert(error, 0, values=y, axis=1) #将正确答案插入第一行
        right = sum(int(x == y) for (x, y) in test_results)
        #打印出用全部测试集进行测试得到的结果
        print( "TrainTest : {0} / {1} = {2}".format(
            right, len(test_data), round(right/len(test_data),6) ))
        return error
        

    def predict(self, X):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        training_inputs = [np.reshape(i, (len(X[0]), 1)) for i in X] # 将一维数组变为二维矩阵[14,1]
        test_results = [np.argmax(self.feedforward(i)) #argmax 返回最大数的索引
                        for i in training_inputs]
        return np.array(test_results) #转换为数组输出

    def data_zip(self, X, y):
        training_inputs = [np.reshape(i, (len(X[0]), 1)) for i in X] # 将一维数组变为二维矩阵[14,1]
        training_results = [vectorized_result(i) for i in y] #转换为二维矩阵，因为最终输出节点为2
        training_data = list(zip(training_inputs, training_results))
        return training_data

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def wb_load(self):
        ''' 读取权重和基值 '''
        fp = open("Titanic/data/wb.pkl", "rb")
        self.weights, self.biases = pickle.load(fp)
        fp.close() 
        return self.weights, self.biases

    def wb_save(self):
        ''' 保存权重和基值并返回 '''
        with open("Titanic/data/wb.pkl", "wb") as fp:
            pickle.dump((self.weights,self.biases), fp, pickle.HIGHEST_PROTOCOL)
            fp.close()
        return self.weights, self.biases

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...1) into a corresponding desired output from the neural
    network."""
    i = int(j)
    e = np.zeros((2, 1))
    e[i] = 1.0
    return e

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function.
       sigmod函数的一阶导数"""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))

def tanh(z):
    return 2*sigmoid(2*z)-1

def relu(z):
    ''' 效果相当差 '''
    x = z
    j = 0 #设置行标，从0开始
    for i in z:
        if i[0] > 0:
            pass # 不变
        else:
            i[0] = 0.001*i[0]
            x[j] = i #leaky ReLU 0.001x
        j = j+1
    return x

def relu_prime(z):
    ''' ReLU导数(分段)：
        x <= 0时，导数为0
        x > 0时，导数为1'''
    x = z
    j = 0 #设置行标，从0开始
    for i in z:
        if i[0] > 0:
            i[0] = 1
            x[j] = i
        else:
            i[0] = 0.001
            x[j] = i #leaky ReLU 0.001x
        j = j+1
    return x
    



'''softplus的导数刚好是sigmoid：
g'(x) = e^x/(e^x+1) = 1/(1+e^-x)'''