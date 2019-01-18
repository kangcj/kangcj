#coding:UTF-8
#搭建卷积神经网络完整框架实例（内含卷积层 采样层 全连接层）

import numpy as np 
import Activators
import CNN
import FCN
from API import *


class ConvertNetwork(object):
	def __init__(self):
		#卷积：图片输入高度，图片输入宽度，通道数，滤波器高度，滤波器宽度，滤波器数目，补0圈数，步长，激活器，学习率
		c_1_h, c_1_w = 256, 256
		self.conv_1 = CNN.Convlayer(c_1_h, c_1_w, 1, 5, 5, 6, 0, 1, Activators.ReluActivator(), 0.01)


		#采样：输入的高度，输入的宽度，通道数，滤波器的高度，滤波器的宽度，步长
		#计算这一层输入的大小
		pl_1_h = int(CNN.Convlayer.calOutputSize(c_1_h, 5, 0, 1))
		pl_1_w = int(CNN.Convlayer.calOutputSize(c_1_w, 5, 0, 1))
		print('pl_1_h, pl_1_w: ', pl_1_h, pl_1_w)
		self.pl_1 = CNN.Maxpooling(pl_1_h, pl_1_w, 6, 2, 2, 2)

		#卷积
		#计算这一层输入的大小
		c_2_h = int(CNN.Convlayer.calOutputSize(pl_1_h, 2, 0, 2))
		c_2_w = int(CNN.Convlayer.calOutputSize(pl_1_w, 2, 0, 2))
		print('c_2_h, c_2_w: ', c_2_h, c_2_w)
		self.conv_2 = CNN.Convlayer(c_2_h, c_2_w, 6, 5, 5, 12, 0, 1, Activators.ReluActivator(), 0.02)
 
		#采样
		pl_2_h = int(CNN.Convlayer.calOutputSize(c_2_h, 5, 0, 1))
		pl_2_w = int(CNN.Convlayer.calOutputSize(c_2_w, 5, 0, 1))
		print('pl_2_h, pl_2_w: ', pl_2_h, pl_2_w)
		self.pl_2 = CNN.Maxpooling(pl_2_h, pl_2_w, 12, 2, 2, 2)

		#全连接:输入大小，输出大小，激活函数，学习率
		f_1_h = int(CNN.Convlayer.calOutputSize(pl_2_h, 2, 0, 2))
		f_1_w = int(CNN.Convlayer.calOutputSize(pl_2_w, 2, 0, 2))
		print('f_1_h, f_1_w: ', f_1_h, f_1_w)
		#全连接层的输入个数(长 * 宽 * 深度)
		f_1_n = int(f_1_h * f_1_w * 12) 
		self.f_1 = FCN.FullConnectedLayer(f_1_n, 18, Activators.SigmoidActivator(), 0.02)


	def forward(self, input_data):
		'''
		向前传播
		input_data 输入图片数据
		'''
		#print('向前传播')
		#卷积
		self.conv_1.forward(input_data)

		#采样
		#print(self.conv_1.output_array.shape)
		self.pl_1.forward(self.conv_1.output_array)

		#卷积
		self.conv_2.forward(self.pl_1.output_array)

		#采样
		self.pl_2.forward(self.conv_2.output_array)

		#全连接层
		f_1_input = self.pl_2.output_array.flatten().reshape((-1, 1))
		self.f_1.forward(f_1_input)


	def backward(self, input_data, labels):
		'''
		反向传播误差
		input_data: 输入图片数据
		labels: 图片对应标签
		'''
		#print('反向传播误差')
		#最后一层的误差
		delta = np.multiply(self.f_1.activator.backward(self.f_1.output), (labels - self.f_1.output))
		
		#全连接层向前传递
		self.f_1.backward(delta)
		self.f_1.update()

		#采样传递误差
		sensitivity_map = self.f_1.delta.reshape(self.pl_2.output_array.shape)
		self.pl_2.backward(self.conv_2.output_array, sensitivity_map)

		#卷积传递误差
		self.conv_2.backward(self.pl_1.output_array, self.pl_2.delta_array, Activators.ReluActivator())
		self.conv_2.update()

		#采样传递误差
		self.pl_1.backward(self.conv_1.output_array, self.conv_2.delta_array)

		#卷积传递误差
		self.conv_1.backward(input_data, self.pl_1.delta_array, Activators.ReluActivator())
		self.conv_1.update()


if __name__ == '__main__':
	data, labels = getTrainingData('C:\\Users\\VISSanKCJ\\Desktop\\histogram')
	network = ConvertNetwork()

	print('TrainingData...')
	max_iter = 5
	for i in range(max_iter):
		print('iteration: ', i + 1)
		for k in range(len(data)): 
			# if k % 10 == 0:
			# 	print(k + 1,'/', len(data))
			input_data = data[k]
			input_data = np.array([input_data])
			result = network.forward(input_data)
			label = labels[k].reshape(-1, 1)
			network.backward(input_data, label)

	np.save("model.npy",network)

	print('Testing...')
	test_data, test_labels = getTrainingData('C:\\Users\\VISSanKCJ\\Desktop\\1')
	right = 0
	for i in range(test_data.shape[0]):
		input_data = np.array([test_data[i]])
		res = network.forward(input_data)
		label = test_labels[i].reshape(-1, 1)#将其变成一列的数组
		prediction = np.argmax(res)
		real = np.argmax(label)
		print('real age: ', real, 'prediction age: ', prediction)
		if prediction == real:
			right += 1
	print('correct ratio: ', right / test_data.shape[0])




