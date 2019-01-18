import numpy as np
import matplotlib.pyplot as plt

#定义输入数据
X = np.array([[1,3,3],
	[1,4,3],
	[1,1,1],
	[1,2,1]])

#定义标签
T = np.array([[1],
	[1],
	[-1],
	[-1]])

#权值初始化
W = np.random.random([3,1])

#学习率设置
lr = 0.1

#神经网络输出
Y = 0

def train():
	global X, Y, W,lr, T
	Y = np.sign(np.dot(X, W))
	E = T - Y
	W += lr * (X.T.dot(E)) / X.shape[0]

for i in range(100):
	train()
	print('epoch:', i+1)
	print('wights:', W)
	Y = np.sign(np.dot(X, W))
	# all()表示Y中的所有值跟T中的所有值都对应相等，才为真  
	# 因为这里的激活函数值只有两个亚目，额1要么-1，在线性神经网络里就不能这么做
	# 线性神经网络的激活函数为 Y = X  ,结果是连续不定的
	if (Y==T).all():
		print('Finished!')
		break


#画图
#正样本的xy坐标
x1 = [3,4]
y1 = [3,3]
#负样本的坐标
x2 = [1,2]
y2 = [1,1]
#定义分类边界线的斜率
k = -W[1] / W[2]
b = -W[0] / W[2]
#通过两点来确定一条直线，用红色的线来画出分界线
x = (0, 5)
plt.plot(x, x * k + b, 'r')
# 用蓝色的点画正样本
plt.scatter(x1, y1, c='b')
# 用黄色的点画负样本
plt.scatter(x2, y2, c='y')
plt.show()