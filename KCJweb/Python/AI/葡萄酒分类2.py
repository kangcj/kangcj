from sklearn.neural_network import MLPClassifier   #从这个封装里导入神经网络这个类
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 

#载入数据
data = np.genfromtxt('C:/Users/VISSanKCJ/Desktop/wine_data.csv', delimiter=',')
x_data = data[:,1:]
y_data = data[:,0]
print(x_data.shape)
print(y_data.shape)

#数据切分
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

#数据标准化 (变成0附近的数值)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#建模
mlp = MLPClassifier(hidden_layer_sizes=(100,60), max_iter=500)
#训练
mlp.fit(x_train, y_train)

#评估
predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))