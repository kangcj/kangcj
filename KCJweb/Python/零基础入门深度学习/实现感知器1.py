#使用书本的框架实现感知器

from functools import reduce
import numpy as np

class Perceptron(object):
    def __init__(self, inputnodes, bias, learning_rate):
        self.inodes = inputnodes
        self.b = bias
        self.lr = learning_rate
        # 权重向量初始化为0
        self.W = [0.0 for _ in range(inputnodes)]

        self.activation_function = lambda x: 1 if x > 0 else 0 

    def __str__(self):
        return 'self.W\t:%s\nself.b\t:%f\n' %(self.W, self.b)

    def train(self, input, target):
        output = self.activation_function(reduce(lambda a, b: a + b, map(lambda x, w: x * w,  input, self.W), 0.0) + self.b)
        output_error = target - output

        self.W += self.lr * output_error * np.array(input)  #不用numpy是不对的
        self.b += self.lr * output_error

    def query(self, input):
        return self.activation_function(reduce(lambda a, b: a + b, map(lambda x, w: x * w,  input, self.W), 0.0) + self.b)



# inputnodes = 2
# bias = 0
# learning_rate = 0.1

n = Perceptron(2, 0, 0.1)

input_data = [[0,0], [0,1], [1,0], [1,1]]
target = [0, 0, 0, 1]

#训练
epoch = 5
for i in range(epoch):
    for k,v in enumerate(input_data):
        n.train(v, target[k])

#测试
print(n.__str__())
print ('1 and 1 = %d' % n.query([1, 1]))
print ('0 and 0 = %d' % n.query([0, 0]))
print ('1 and 0 = %d' % n.query([1, 0]))
print ('0 and 1 = %d' % n.query([0, 1]))




