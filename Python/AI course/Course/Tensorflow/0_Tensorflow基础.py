# encoding:UTF-8
#2019-1-17
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt



# '''
# 第一节
# 创建会话，启动会话
# '''
# #创建两个常量
# m1 = tf.constant([[1,2]])
# m2 = tf.constant([[1], [2]])

# #矩阵乘法
# Product = tf.matmul(m1, m2)

# #定义会话方式1
# sess = tf.Session()
# #调用sess中run方法来执行矩阵乘法
# result = sess.run(Product)
# print(result)
# sess.close()


# #定义会话方式2
# with tf.Session() as sess:
# 	result = sess.run(Product)
# 	print(result)



# '''
# 第二节
# 变量的使用
# '''
# #定义一个变量
# x1 = tf.Variable([1, 2])
# #定义一个常量
# x2 = tf.constant([3, 3])
# #减法op
# sub = tf.subtract(x1, x2)
# #加法op
# add = tf.add(x1, sub)

# #定义了变量就需要所有变量初始化，常量不用.但还是需要通过下面的会话才会运行
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	#执行变量初始化
# 	sess.run(init)
# 	print(sess.run(sub))
# 	print(sess.run(add))


# '''
# 第三节
# Fetch和Feed的用法
# '''
# #Fetch:可以在会话中同时计算多个tensor或执行多个操作
# #定义三个常量
# input1 = tf.constant(3.0)
# input2 = tf.constant(4.0)
# input3 = tf.constant(5.0)
# #加法op
# add = tf.add(input1, input2)
# #乘法op
# mul = tf.multiply(input3, add)

# with tf.Session() as sess:
# 	result1, result2 = sess.run([mul, add])#用方括号括起来可以同时计算多个数值
# 	print(result1, result2)


# #Feed:先定义占位符，等需要的时候再传入数据
# input1 = tf.placeholder(tf.float32)#并没有实际的数值，用占位符传入了数值的类型，先让流程前进
# input2 = tf.placeholder(tf.float32)
# mul = tf.multiply(input1, input2)

# with tf.Session() as sess:
# 	print(sess.run(mul, feed_dict={input1: 8.0, input2: 9.0}))#传入的时候需要以字典的形式传递



'''
第四节
线性回归
'''
# import numpy as np 
# import tensorflow as tf 
# import matplotlib.pyplot as plt 

# x_data = np.random.rand(100)
# noise = np.random.normal(0, 0.01, x_data.shape)#生成一些噪声
# y_data = x_data * 0.1 + 0.2 + noise#真实值

# # plt.scatter(x_data, y_data)
# # plt.show()


# #构建一个线性模型
# d = tf.Variable(np.random.rand(1))
# k = tf.Variable(np.random.rand(1))
# y = k * x_data + d #模型的预测 值

# #二次代价函数
# loss = tf.losses.mean_squared_error(y_data, y)#tf已经封装好了均方差

# #定义一个梯度下降法优化器
# optimizer = tf.train.GradientDescentOptimizer(0.3)

# #最小化代价函数
# train = optimizer.minimize(loss)

# #初始化变量
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)
# 	for i in range(201):
# 		sess.run(train)
# 		if i % 20 == 0:
# 			print(i, sess.run([k, d]))
# 	y_pred = sess.run(y)
# 	plt.scatter(x_data, y_data)
# 	plt.plot(x_data, y_pred, 'r-', lw = 3)
# 	plt.show()




'''
第五节
非线性回归
'''

# '''
# 准备数据
# '''
# #Numpy生成200个随机点
# x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] #在列上将升上了一个维度，即生成了200行1列的数组
# #x_data = np.linspace(-0.5, 0.5, 200).reshape(-1, 1)#一样可将其变成一列
# noise = np.random.normal(0, 0.02, x_data.shape)
# y_data = np.square(x_data) + noise


# #assert  False#断言语句，让其运行此处报错停止，可以用来检查错误
# # plt.scatter(x_data, y_data)
# # plt.show()


# '''
# 神经网络
# tf的计算流程
'''
# #定义两个placeholder
# x = tf.placeholder(tf.float32, [None, 1])#n行1列
# y = tf.placeholder(tf.float32, [None, 1])

# #神经网络结构： 1-20-1
# w1 = tf.Variable(tf.random_normal([1, 20]))
# b1 = tf.Variable(tf.zeros([20]))#一般初始化偏置时为0或者一个常数
# xw_plus_b1 = tf.matmul(x, w1) + b1
# l1 = tf.nn.tanh(xw_plus_b1)

# w2 = tf.Variable(tf.random_normal([20, 1]))
# b2 = tf.Variable(tf.zeros([1]))
# xw_plus_b2 = tf.matmul(l1, w2) + b2
# prediction = tf.nn.tanh(xw_plus_b2)


# #二次代价函数
# loss = tf.losses.mean_squared_error(y, prediction)
# #使用梯度下降法最下化loss
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# '''
# 会话计算
# '''
# with tf.Session() as sess:
# 	#变量初始化
# 	sess.run(tf.global  
# _variables_initializer())
# 	for _ in range(3000):
# 		sess.run(train, feed_dict = {x:x_data, y:y_data})

# 	#获得预测值
# 	prediction_value = sess.run(prediction, feed_dict = {x:x_data})
# 	#画图
# 	plt.scatter(x_data, y_data)
# 	plt.plot(x_data, prediction_value, 'r-', lw=5)
# 	plt.show()
