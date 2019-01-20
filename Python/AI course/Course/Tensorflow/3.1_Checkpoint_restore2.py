# coding: utf-8
# 2019-1-20

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
# 定义批次大小
batch_size = 64
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
# 这里的模型参数需要跟之前训练好的模型参数一样
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 交叉熵代价函数
loss = tf.losses.softmax_cross_entropy(y,prediction)
# 使用Adam优化器
train = tf.train.AdamOptimizer(0.001).minimize(loss)

# 定义saver用于载入模型
# max_to_keep=5,在指定路径下最多保留5个模型，超过5个模型就会删除老的模型
saver = tf.train.Saver(max_to_keep=5)

# 定义会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    
    # 载入训练好的参数
    saver.restore(sess,'models/my_model.ckpt')
    # 再次计算测试集准确率
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    # 在原来模型的基础上再训练11个周期
    for epoch in range(11):
        for batch in range(n_batch):
            # 获取一个批次的数据和标签
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            # 训练模型
            sess.run(train, feed_dict={x:batch_xs,y:batch_ys})
        # 计算测试集准确率
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        # 打印信息
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
        # 保存模型，global_step可以用来表示模型的训练次数或者训练周期数
        saver.save(sess,'models/my_model.ckpt', global_step=epoch)






