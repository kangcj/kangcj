# coding: utf-8
# 2019-1-22

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot='True')

# 输入图片是28*28
n_inputs = 28 # 输入一行，一行有28个数据
max_time = 28 # 一共28行
lstm_size = 100 # 隐藏层单元
n_classes = 10 # 10个分类
batch_size = 64 # 每批次64个样本
n_batch = mnist.train.num_examples // batch_size #计算一共有多少批次


# 这里的None表示第一维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.truncated_normal([lstm_size, n_classes]))
b = tf.V ariable(tf.constant(0.1, shape=[n_classes]))

# 定义RNN网络
def RNN(x, w, b):
	inputs = tf.reshape(x, [-1, max_time, n_inputs])
	# 定义LSTM
	lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
#    final_state[state, batch_size, cell.state_size]
#    final_state[0]是cell state
#    final_state[1]是hidden_state
#    outputs: The RNN output `Tensor`.
#       If time_major == False (default), this will be a `Tensor` shaped:
#         `[batch_size, max_time, cell.output_size]`.
#       If time_major == True, this will be a `Tensor` shaped:
#         `[max_time, batch_size, cell.output_size]`.
	outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
	results = tf.nn.softmax(tf.matmul(final_state[1], w) + b)
	return results


# 计算RNN的返回结果
prediction = RNN(x, W, b)
loss = tf.losses.softmax_cross_entropy(y, prediction)
train = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(11):
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train, feed_dict={x:batch_xs, y:batch_ys})
		acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
		print("Iter ", str(epoch+1), '  Test Accuracy ', str(acc))

