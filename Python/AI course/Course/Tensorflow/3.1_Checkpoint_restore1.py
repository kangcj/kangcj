# coding: utf-8
# 2019-1-20

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 定义批次大小
batch_size = 64
# 计算一共多少批次
n_batch = mnist.train.num_examples // batch_size

with tf.Session() as sess:

	# 载入模型结构
	saver = tf.train.import_meta_graph('models/my_model.ckpt.meta')
	# 载入模型参数
	saver.restore(sess, 'models/my_model.ckpt')
	# 根据tensor的名字获取对应的tensor
	# 之前保存模型的时候模型输出保存为Output, ':0'是保存模型参数时自动加上的，所以这里也要写上
	output = sess.graph.get_tensor_by_name('output:0')
	accuracy = sess.graph.get_tensor_by_name('accuracy:0')
	# 之前保存模型的时候模型训练保存为train，注意这里的train是operation不是tensor
	train = sess.graph.get_operation_by_name('train')

	# 把测试集喂到网络中计算准确率
	# x_input和y_input是模型数据和标签的输入，':0'是保存模型参数时自动加上的，所以这里也要写上
	print(sess.run(accuracy, feed_dict={'x_input:0':mnist.test.images, 'y_input:0':mnist.test.labels}))

	# 继续训练原模型11个周期
	for epoch in range(11):
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train, feed_dict={'x_input:0':batch_xs, 'y_input:0':batch_ys})
		# 计算测试集准确率
		acc = sess.run(accuracy, feed_dict={'x_input:0':mnist.test.images, 'y_input:0':mnist.test.labels})
		# 打印信息
		print("Iter " + str(epoch + 1) + ",Testing Accuracy " + str(acc))

