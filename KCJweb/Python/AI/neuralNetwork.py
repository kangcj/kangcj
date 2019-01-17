import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class nueralNetwork():
	#3层神经网络

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		#神经网络基本框架
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes

		self.lr = learning_rate

		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		self.activation_function = lambda x: scipy.special.expit(x)

		pass

	def train(self, inputs_list, targets_list):
		#训练神经网络，更新权值矩阵
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		#由输入到输出
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		output_inputs = np.dot(self.who, hidden_outputs)
		output_outputs = self.activation_function(output_inputs)

		#反向传递误差
		output_errors = targets - output_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		#跟新权值矩阵
		self.who += self.lr * np.dot(output_errors * output_outputs * (1 - output_outputs), np.transpose(hidden_outputs))
		self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

		pass

	def query(self, input_list):
		#测试神经网络
		inputs = np.array(input_list, ndmin=2).T

		hidden_inputs = np.dot(wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		output_inputs = np.dot(whd. hidden_outputs)
		output_outputs = self.activation_function(output_inputs)

		return output_outputs




a, s, d, f = 784, 100, 10, 0.3

n = nueralNetwork(a, s, d, f)

#读取训练数据
training_data_file = open('C:/Users/VISSanKCJ/Desktop/data/mnist_train_100.csv')
training_data_list = training_data_file.readlines()
training_data_file.close

#训练数据
i = 0
for record in training_data_list:

	all_values = record.split(',')

	inputs = (np.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
	targets = np.zeros(d) + 0.01
	targets[int(all_values[0])] = 0.99

	n.train(inputs, targets)

	i += 1

	if i<=99:
		print("训练第%d次" %i)
	else:
		print("训练第100次\n100次训练完毕。")

	pass






