from functools import reduce

class Perceptron():
	def __init__(self, input_num, activator):
		self.activator = activator
		#权重向量初始化为0
		self.weights = [0.0 for _ in range(input_num)]
		#偏置项初始化为0
		self.bias = 0.0

	def __str__(self):
		#打印学习到的权重和偏置项
		return 'weights\t:%s\nbias\t:%f\n' %(self.weights, self.bias)

	def predict(slef, input_vec):
		#输入向量，输出感知器的结果
		#把input_vec[x1,x2,x3,...]和wights[w1,w2,w3,...]打包在一起
		#变成[(x1,w1),(x2,w2),(x3,w3),...]
		#然后利用map函数计算[x1*w1,x2*w2,x3*w3]
		#最后利用reduce求和
		return self.activator(reduce(lambda a, b: a + b,map(lambda x, w: x * w, 
			zip(input_vec, self.weights)), 0.0) + self.bias)

	def train(self, input_vecs, labels, iteration, rate):
		for i in range(iteration):
			self._one_iteration(input_vecs, labels, rate)

	def _one_iteration(self, input_vecs, labels, rate):
		#一次迭代把所有训练数据都过一遍
		#把输入和输出打包在一起，成为样本的列表[(input_vec,label),...]
		#而每个训练样本是(input_vec, labels)
		samples = zip(input_vecs, labels)
		#对每个样本。按照感知器规则更新权重
		for (input_vec, label) in  samples:
			output = self.predict(input_vec)
			#更新权重
			self._update_weights(self, input_vec, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		#按照感知器规则更新权重
		delta = label - output
		self.weights = map(
			lambda (x, w): w + rate * delta * x,
			zip(input_vec, self.weights))
		#更新bias
		self.bias += rate * delta



#接下来我们用这个感知器来实现and函数

def f(x):
	...
	定义激活函数f
	...
	return 1 if x > 0 else 0

def get_training_dataset():
	...
	基于and真值表构建训练数据
	...
	input_vecs = [[1,1], [0,0], [1,0], [0,1]]
	labels = [1, 0, 0, 0]
	return input_vecs, labels

def train_and_perceptron():
	...
	使用and真值表训练感知器
	...
	p = Perceptron(2, f)
	input_vecs, lables = get_training_dataset()
	p.train(input_vecs, labels, 10, 0.1)
	return p


if __name__=='__main__':
	and_perceptron = train_and_perceptron()
	print(and_perceptron)
	#测试
	print '1 and 1 =%d' %  and_perceptron.predict([1,1])