# coding: utf-8
# 2019-1-19


import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 64
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

# 定义3个Placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 神经网络 784-1000--500-10
W1 = tf.Variable(tf.truncated_normal([784, 1000], stddev=0.1))# stddev 为标准差
b1 = tf.Variable(tf.zeros([1000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([1000, 500]))
b2 = tf.Variable(tf.zeros([500]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([500, 10]))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# l2正则项(可选l1正则项)
l2_loss = tf.nn.l2_loss(W1) +tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3)

# 交叉熵
# loss = tf.losses.softmax_cross_entropy(y, prediction)
loss = tf.losses.softmax_cross_entropy(y, prediction) + 0.0005 * l2_loss# o.ooo5为正则化系数
# 使用梯度下降法
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 结果放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# 先将布尔型转换为浮点型1.0或者0.0，然后再求数组的平均值

with tf.Session() as sess:
	# 初始化变量
	sess.run(tf.global_variables_initializer())
	for epoch in range(31):
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.5})
		test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
		train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
		print('Iter: ' , str(epoch + 1),'  Testing Accuracy: ', str(test_acc),'  Training Accuracy: ', str(train_acc) )



'''
Dropout，正则化起到一定过拟合作用
Dropout，正则化不是所有时候都适用，当遇到比较复杂的模型，神经网络层数较多时适合使用。
测试时keep_drop设置为1.0, 训练时keep_drop设置为0.5表示不是所有神经元参与计算
'''