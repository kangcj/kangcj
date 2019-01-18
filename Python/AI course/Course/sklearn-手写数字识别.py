from sklearn.neural_network import MLPClassifier   #从这个封装里导入神经网络这个类
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

digits = load_digits()
x_data = digits.data
y_data = digits.target
print(x_data.shape)
print(y_data.shape)

# plt.imshow(digits.images[3], cmap='gray')
# plt.show()

#数据拆分
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

#构建模型，2个隐藏层。第一个隐藏层100个神经元，第二个隐藏层50个神经元，训练500周期
mlp = MLPClassifier(hidden_layer_sizes=(100, 60), max_iter=500)
mlp.fit(x_train, y_train)

predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions)) #测试集标签和预测值对比