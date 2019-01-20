# coding: utf-8
# 2019-1-20

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 64
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# 卷积层
#x input tensor of shape `[batch, in_height, in_width, in_channels]`
#W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
#`strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
#padding: A `string` from: `"SAME", "VALID"`
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 池化层
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 改变x的格式转为4D的格式[batch, in_height, in_width, in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5, 5, 1, 32]) # 5X5的卷积窗口，输入通道数是1，输出通道数是32
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
pool1 = max_pool_2x2(conv1) # 进行Max-pooling


# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
pool2 = max_pool_2x2(conv2)


# 把池化层2的输出平化为1维
pool2_flat = tf.reshape(pool2, [-1, 7*7*64])


# 初始化第一个全连接层 7*7*64-1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) +  b_fc1)
# keep_prob 表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)


# 初始化第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) +  b_fc2)


# 交叉熵代价函数，使用AdamOptimizer优化
loss = tf.losses.softmax_cross_entropy(y, prediction)
train = tf.train.AdamOptimizer(0.0001).minimize(loss)


# 求准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(21):
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
		acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
		print('Iter ', str(epoch+1), '  Testing accuracy ', str(acc) )



