# coding: utf-8
#2019-1-18


import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#批次大小
batch_size = 64
#计算一个周期有多少批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#创建一个简单的神经网络 784-800-10
# W = tf.Variable(tf.random_normal([784, 10]))
# b = tf.Variable(tf.zeros([10]))
W1 = tf.Variable(tf.truncated_normal([784,800], stddev=0.1))
b1 = tf.Variable(tf.zeros([800]) + 0.1)
xw_plus_b1 = tf.matmul(x, W1) + b1
y1 = tf.nn.relu(xw_plus_b1)

W2 = tf.Variable(tf.truncated_normal([800, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]) + 0.1)
xw_plus_b2 = tf.matmul(y1, W2) + b2
prediction = tf.nn.softmax(xw_plus_b2) 

#二次代价函数
#loss = tf.losses.mean_squared_error(y, prediction)
loss = tf.losses.softmax_cross_entropy(y, prediction)
#使用梯度下降法
#train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
train = tf.train.AdamOptimizer(0.001).minimize(loss)

#结果存放在一个布尔型列表中tf.argmax(prediction, 1)为在第一个维度下最大值所在索引
#[None,2]其中None在数组中为第一个维度。tf.argmax(prediction, 1)为在第一个维度下最大值所在索引。最终correct_prediction为一个布尔型列表
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# print(sess.run(y2.shape))
	# assert False
	for epoch in range(20):
		for batch in range(n_batch):
			#获取一个批次的数据和标签
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train, feed_dict = {x:batch_xs, y:batch_ys})
		#每训练一个周期做一次测试
		acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
		print('Iteration: ', str(epoch + 1), '	Testing accuracy: ', str(acc))



'''
不同函数以及参数设置会影响训练结果

权值矩阵初始化一般比较常用的是截断正态分布的随机函数 W = tf.Variable(tf.truncated_normal([784,10], stddev=0.1))；
在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。

偏置一般比较常用的为0或0.1 b = tf.Variable(tf.zeros([10]) + 0.1)

学习率和周期设置大一点也可以提升学习效率

tf激活函数：tf.nn.sigmoid()，tf.nn.tanh()；tf.nn.relu()

分类模型一般用交叉熵代价函数，回归模型一般用二次代价函数。
tf中交叉熵代价函数有tf.losses.sigmoid_cross_entropy();tf.losses.softmax_cross_entropy(),具体用哪一种取决于你最后用的什么激活函数。

优化器一般优先选择AdamOptimizer,且此时学习率一般是指比较小为0.001或者0.0001
'''