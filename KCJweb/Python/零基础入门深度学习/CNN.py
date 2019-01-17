#coding=utf-8
#2019-1-16
import numpy as np 

from Activators import ReluActivator, IdentityActivator

#获取卷积区域
#ij为卷积后单个输出值的下标
def get_patch(input_array, i, j, filter_w,
				filter_h, stride):
	'''
	从输入数组中获取本次卷积的区域
	自动适配输入为2D和3D的情况
	'''
	start_i = i * stride
	start_j = j * stride
	if input_array.ndim == 2:
		return input_array[start_i:start_i + filter_h,start_j:start_j + filter_w]
	elif input_array.ndim == 3:
		return input_array[:,start_i:start_i + filter_h,start_j:start_j + filter_w]


#获取一个2D区域的最大值所在的索引
def getMax_index(array):
	max_i = 0
	max_j = 0
	maxValue = array[0][0] 
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			if array[i][j] > maxValue:
				maxValue = array[i][j]
				max_i = i
				max_j = j
	return max_i, max_j


#计算卷积
def conv(input_array,
			kernel_array,
			output_array,
			stride, b):
	'''
	计算卷积
	自动适配输入为2D和3D的情况
	'''
	channel_number = input_array.ndim
	output_w = output_array.shape[1]
	output_h = output_array.shape[0]
	kernel_w = kernel_array.shape[-1]
	kernel_h = kernel_array.shape[-2]
	for i in range(output_h):
		for j in range(output_w):
			output_array[i][j] = (
				get_patch(input_array, i, j,
					kernel_w, kernel_h,stride) * kernel_array
				).sum() + b

#为数组增加Zero padding
def addZero(input_array, n):
	'''
	为数组增加ZeroPadding
	自动适配输入为2D和3D的情况
	'''
	if n == 0:
		return input_array
	else:
		if input_array.ndim == 3:
			input_d = input_array.shape[0]
			input_h = input_array.shape[1]
			input_w = input_array.shape[2]
			added_array = np.zeros((
				input_d,
				input_h + 2 * n,
				input_w + 2 * n))
			added_array[:,
				n : n + input_h,
				n : n + input_w] = input_array
		elif input_array.ndim == 2:
			input_h = input_array[0]
			input_w = input_array[1]
			added_array = np.zeros((
				input_h + 2 * n,
				input_w + 2 * n))
			added_array[n : n + input_h,
				n : n + input_w] = input_array

		return added_array

#对numpy数组每个元素进行激活函数操作
def elementActivation(array, function):
	for i in np.nditer(array, op_flags=['readwrite']):
		i[...] = function(i)



class Filter():
	def __init__(self, height, width, depth):
		self.W = np.random.uniform(-1e-4, 1e-4,
			(depth, height, width))
		self.b = 0
		self.W_grad = np.zeros(self.W.shape)
		self.b_grad = 0

	def __repr__(self):
		return 'filter weights:\n%s\nbias:\n%s' % (
			self.W, self.b)

	def get_weights(self):
		return self.W

	def get_bias(self):
		return self.b

	def update(self, lr):
		self.W -= lr * self.W_grad
		self.b -= lr * self.b_grad


class Convlayer():
	def __init__(self, input_h, input_w, channel_number,
				filter_h, filter_w, filter_number,
				zerolayer_n, stride, activator,
				lr):
		self.input_h = input_h
		self.input_w = input_w
		self.channel_n = channel_number
		self.filter_h = filter_h
		self.filter_w = filter_w
		self.filter_n = filter_number
		self.zero_n = zerolayer_n
		self.stride = stride

		self.output_h = Convlayer.calOutputSize(self.input_h, self.filter_h, self.zero_n, self.stride)
		self.output_w = Convlayer.calOutputSize(self.input_w, self.filter_w, self.zero_n, self.stride)
		self.output_array = np.zeros((self.filter_n,
			self.output_h, self.output_w))

		self.filters = []
		for i in range(self.filter_n):
			self.filters.append(Filter(filter_h, filter_w,
				channel_number))

		self.activator = activator
		self.lr = lr

	def forward(self, input_array):
		'''
		计算卷积层的输出，输出结果保存在self.output_array中
		'''
		self.input_array = input_array
		self.added_input_array = addZero(input_array, self.zero_n)
		for f in range(self.filter_n):
			filter = self.filters[f]
			conv(self.added_input_array, filter.get_weights(),
				self.output_array[f], self.stride, filter.get_bias())
		elementActivation(self.output_array,
							self.activator.forward)

	def backward(self, input_array, sensitivity_array,
					activator):
		'''
		计算传递给前一层的误差项，以及计算的每个权重的梯度
		前一层的误差项保存在self.delta_array
		梯度保存在Filter对象的self.weights_frad中
		'''
		self.forward(input_array)
		self.bp_sensitivity_map(sensitivity_array, activator)
		self.bp_gradient(sensitivity_array)

	def update(self):
		'''
		梯度下降，跟新权重
		'''
		for filter in self.filters:
			filter.update(self.lr)


	def bp_sensitivity_map(self, sensitivity_array,
							activator):
		'''
		计算传递到上一层的sensitivity map
		sensitivity_array: 本层的sensitivity map
		activator: 上一层的激活函数
		'''
		#处理卷积步长，对原始sensitivity map 进行扩展
		expanded_array = self.expand_Sensitivity_Map(sensitivity_array)
		#full卷积，对sensitivity map进行addZero
		#虽然原始输入的zero单元也会获得残差
		#但这个残差不需要继续向上传递，因此就不计算了
		expanded_w = expanded_array.shape[2]
		zerolayer_n = int((self.input_w + self.filter_w - 1 - expanded_w) / 2)
		added_array = addZero(expanded_array, zerolayer_n)
		#初始化delta_array,用于保存传递到上一层的sensitivity map
		self.delta_array = self.create_delta_array()
		#对于具有多个filter的卷积层来说，最终传递到上一层sensitivity map
		#相当于所有的filter的sensitivity map之和
		for f in range(self.filter_n):
			filters = self.filters[f]
			#print(filters.get_weights())
			#将filter权重翻转180度
			# flipped_weights= np.array(map(
			# 	lambda i: np.rot90(i, 2),
			# 	filters.get_weights()))
			flipped_weights = []
			for w in filters.get_weights():
				flipped_weights.append(np.rot90(w, 2))
			flipped_weights = np.array(flipped_weights)
			#print(flipped_weights)
			#计算与一个filter对应的delta_array
			delta_array = self.create_delta_array()
			for d in range(delta_array.shape[0]):
				# print(added_array.shape, flipped_weights.shape, delta_array.shape)
				# print(added_array[f].shape, flipped_weights[d].shape, delta_array[d].shape)
				conv(added_array[f], flipped_weights[d],
					delta_array[d], 1, 0)
			self.delta_array += delta_array
		#将计算结果与激活函数的偏导数做乘法操作
		derivative_array = np.array(self.input_array)
		elementActivation(derivative_array, activator.backward)
		self.delta_array *= derivative_array 

	def expand_Sensitivity_Map(self, sensitivity_array):
		"""
		对步长大于1时的sensitivity map相应位置进行补0，将其还原成步长为1时的sensitivity map,再用（14）[（8）]进行求解
		"""
		depth = sensitivity_array.shape[0]

		# 获得stride=1时的sensitivity map大小
		expanded_w = Convlayer.calOutputSize(self.input_w, self.filter_w, self.zero_n, 1)
		expanded_h = Convlayer.calOutputSize(self.input_h, self.filter_h, self.zero_n, 1)
		expand_array = np.zeros((depth, expanded_h, expanded_w))

		for i in range(self.output_h):
			for j in range(self.output_w):
			    i_pos = i * self.stride
			    j_pos = j * self.stride
			    expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
		return expand_array


	def bp_gradient(self, sensitivity_array):
		"""
        计算偏置项的梯度
        偏置项的梯度就是sensitivity map 所有误差项之和
        """
		expanded_error_array = self.expand_Sensitivity_Map(sensitivity_array)
		for i in range(self.filter_n):
		    filters = self.filters[i]
		    for d in range(filters.W.shape[0]):
		        conv(self.added_input_array[d], expanded_error_array[i], filters.W_grad[d], 1, 0)
		    filters.b_grad = expanded_error_array[i].sum()

	def create_delta_array(self):
		return np.zeros((self.channel_n,
			self.input_h, self.input_w))

	@staticmethod
	def calOutputSize(input_size, fliter_size, zerolayer_n, stride):
		'''
		计算feature map 的大小
		'''
		return int((input_size - fliter_size + 2 * zerolayer_n) / stride + 1)

class Maxpooling():
	def __init__(self, input_h, input_w, channel_number,
				filter_h, filter_w, stride):
		self.input_h = input_h
		self.input_w = input_w
		self.channel_n = channel_number
		self.filter_h = filter_h
		self.filter_w = filter_w
		self.stride = stride

		self.output_h = int((input_h - filter_h) / stride + 1)
		self.output_w = int((input_w - filter_w) / stride + 1)
		self.output_array = np.zeros((self.channel_n,
						self.output_h, self.output_w))

	def forward(self, input_array):
		for d in range(self.channel_n):
			for i in range(self.output_h):
				for j in range(self.output_w):
					#print(self.output_array[d, i, j])
					self.output_array[d, i, j] = get_patch(input_array[d], i, j,
							self.filter_w,
							self.filter_h,
							self.stride).max()
					#print(self.output_array[d, i, j])

	def backward(self, input_array, sensitivity_array):
		self.delta_array = np.zeros(input_array.shape)
		for d in range(self.channel_n):
			for i in range(self.output_h):
				for j in range(self.output_w):
					patch_array = get_patch(
						input_array[d], i, j,
						self.filter_w,
						self.input_h,
						self.stride)
					k, l = getMax_index(patch_array)
					self.delta_array[d,
						i * self.stride + k,
						j * self.stride + l] = \
						sensitivity_array[d, i, j]




	


























	

