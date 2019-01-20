# coding: utf-8
# 2019-1-20

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 每个批次64张照片
batch_size = 64
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
# 给模型数据输入的入口起名为x-input
x = tf.placeholder(tf.float32,[None,784], name='x-input')
# 给模型标签输入的入口起名为y-input
y = tf.placeholder(tf.float32,[None,10], name='y-input')

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
b = tf.Variable(tf.zeros([10])+0.1)
# 给模型输出起名为output
prediction = tf.nn.softmax(tf.matmul(x,W)+b, name='output')

# 交叉熵代价函数
loss = tf.losses.softmax_cross_entropy(y,prediction)
# 使用Adam优化器，给优化器operation起名为train
train = tf.train.AdamOptimizer(0.001).minimize(loss, name='train')

# 最后correct_prediction是一个布尔型的列表
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 给准确率tensor起名为accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 运行11个周期
    for epoch in range(11):
        for batch in range(n_batch):
            # 获取一个批次的数据和标签
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            # 喂到模型中做训练
            sess.run(train, feed_dict={x:batch_xs,y:batch_ys})
        # 每个周期计算一次测试集准确率
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        # 打印信息
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
    # 保存模型参数和结构,把变量变成常量
    # output_node_names设置可以输出的tensor
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output','accuracy'])
    # 保存模型到目录下的models文件夹中
    with tf.gfile.FastGFile('pb_models/my_model.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())