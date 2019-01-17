import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

class neuralNetwork():
    
    
    def __init__(self, inputnodes, hiddennodes, outputnodes,learningrate):
        
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.activation_function = lambda x: ss.expit(x)
         
        pass
    
    def train(self, input_list, target_list):
        
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(inputs))
        
        pass
    
    def query(self, input_list):
        
        inputs = np.array(input_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.2

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes,learning_rate)

training_data_file = open("C:/Users/VISSanKCJ/Desktop/data/mnist_train_100.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

c = 1
for i in range(5):
    print("Iteration: %d" % c)
    c += 1
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass


test_data_file = open("C:/Users/VISSanKCJ/Desktop/data/mnist_test_10.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()



# all_values = test_data_list[5].split(',')
# print(all_values[0])

# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')

# a = n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
# print(a)
# plt.show()

results = []

for record in test_data_list:

    all_values = record.split(',')
    corret_label = int(all_values[0])
    print('corret_label is', corret_label)
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print('network answer is', label)

    if label == corret_label:
        results.append(1)
    else:
        results.append(0)

print(results)

scorecard_array = np.array(results)
corret_rate = scorecard_array.sum() / scorecard_array.size
print('corret_rate is', corret_rate)
        
     