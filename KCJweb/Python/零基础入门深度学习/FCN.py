#coding:UTF-8
#全连接层实现类

import numpy as np 
from Activators import *


class FullConnectedLayer(object):
	'''
	全连接层构造函数
	'''
	def __init__(self, input_size, output_size, activator, learningrate):
		self.input_size = input_size
		self.output_size = output_size
		self.activator = activator
		self.lr = learningrate

		self.W = np.random.normal(-0.1, 0.1, (output_size, input_size))
		self.b = np.zeros((output_size, 1))

		self.output = np.zeros((output_size, 1)) #输出初始化为0向量

	def forward(self, input_array):
		'''
		向前传播
		'''
		self.input = input_array
		self.output = self.activator.forward(np.dot(self.W, self.input) + self.b)

	def backward(self, delta_array):
		'''
		误差反向传递
		'''
		self.delta = np.multiply(self.activator.backward(self.input), np.dot(self.W.T, delta_array))
		self.W_grad = np.dot(delta_array, self.input.T)
		self.b_grad = delta_array

	def update(self):
		'''
		用梯度下降法更新权重
		'''
		self.W += self.lr * self.W_grad
		self.b += self.lr * self.b_grad





