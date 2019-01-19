# coding: utf-8
# 2019-1-19


import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#每个批次64张照片
batch_size = 64
#计算一共多少批次
n_batch = mnist.train.num_examples // batch_size

#定义两个Placeholder
#给模型数据输入的入口起名为x_input
x = tf.placeholder(tf.float32, [None, 784], name='x_input')
#给模型标签输入的入口起名为y_input
y = tf.placeholder(tf.float32, [None, 10], name='y_input')

#神经网络 784-10
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]) + 0.1)
#给模型输出起名为output
prediction = tf.nn.softmax(tf.matmul(x, W) + b, name='output')

#交叉熵代价函数
loss = tf.losses.softmax_cross_entropy(y, prediction)
#使用Adam优化器，给优化器operation起名为train
train = tf.train.AdamOptimizer(0.001).minimize(loss, name='train')

#求准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')


#定义Saver用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(11):
		for batch in range(n_batch):
			#获取一个批次的数据和标签
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			#喂到模型中做训练
			sess.run(accuracy, feed_dict={x: batch_xs, y:batch_ys})
		#每个周期计算一次测试集的准确率
		acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
		#打印信息
		print('Iter ', str(epoch + 1), ' Testing Accuracy ', str(acc))

	#保存模型
	saver.save(sess, 'models/my_model.ckpt')