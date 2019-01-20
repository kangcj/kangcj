# coding: utf-8
# 2019-1-20

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 载入模型
with tf.gfile.FastGFile('pb_models/my_model.pb', 'rb') as f:
    # 创建一个图
    graph_def = tf.GraphDef()
    # 把模型文件载入到图中
    graph_def.ParseFromString(f.read())
    # 载入图到当前环境中
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # 根据tensor的名字获取到对应的tensor
    # 之前保存模型的时候模型输出保存为output，":0"是保存模型参数时自动加上的，所以这里也要写上
    output = sess.graph.get_tensor_by_name('output:0')
    # 根据tensor的名字获取到对应的tensor
    # 之前保存模型的时候准确率计算保存为accuracy，":0"是保存模型参数时自动加上的，所以这里也要写上
    accuracy = sess.graph.get_tensor_by_name('accuracy:0')
    # 预测准确率
    print(sess.run(accuracy,feed_dict={'x-input:0':mnist.test.images,'y-input:0':mnist.test.labels}))