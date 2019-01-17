import numpy as np 
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt 

#载入数据
digits = load_digits()

#显示图片
# plt.imshow(digits.images[1], cmap='gray')
# plt.show()

#数据
X = digits.data 
Y = digits.target
# print(X.shape)
# print(Y.shape)
# print(X[:3])
# print(Y[:3])

#定义一个神经网络，结构：64-100-10
#分别定义输入层到隐藏层与隐藏层到输出层的权值矩阵
V = np.random.random((64,100)) * 2 - 1  #-1到1
W = np.random.random((100,10)) * 2 - 1

#数据切分，3/4为训练集，1/4为测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#标签二值化
#0：1000000000
labels_train = LabelBinarizer().fit_transform(Y_train)
# print(Y_train[:3])
# print(labels_train[:3])


#激活函数
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#激活函数的导数
def dsigmoid(x):
	return x * (1 - x)

#训练模型
def train(X, Y, epoch=10000, lr=0.11):
	global V, W
	for n in range(epoch+1):
		#随机选取一个数据
		i = np.random.randint(X.shape[0])
		#获取一个数据
		x = X[i]
		x = np.atleast_2d(x) #变成2维的，用于做矩阵乘法[[]]
		#BP算法公式
		#计算隐藏层和输出层的输出
		L1 = sigmoid(np.dot(x, V))
		L2 = sigmoid(np.dot(L1, W))
		#计算输出层和隐藏层从输出误差
		L2_delta = (Y[i] - L2) * dsigmoid(L2)
		L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)
		#更新权值
		W += lr * L1.T.dot(L2_delta)
		V += lr * x.T.dot(L1_delta)

		#每训练1000次预测一次准确率
		if n % 1000 == 0:
			output = predict(X_test)
			predictions = np.argmax(output, axis=1)
			# print(predictions)
			acc = np.mean(np.equal(predictions, Y_test)) #如果预测值与标签相等则为真即为1，然后1的个数除以总个数为准确率
			print('epoch:', n, 'accuracy:', acc)

def predict(x):
	L1 = sigmoid(np.dot(x, V))
	L2 = sigmoid(np.dot(L1, W))
	return L2 


train(X_train, labels_train, 30000)