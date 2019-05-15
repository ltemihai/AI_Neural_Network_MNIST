import numpy as np
from random import randint

import file_reader
import neural_network_keras
from neural_network import NeuralNetwork, get_accuracy, error

data_test = file_reader.open_file('mnist_test.csv')
data_train = file_reader.open_file('mnist_train.csv')

training_data = file_reader.normalize_data(data_train)
training_labels = file_reader.normalize_labels(data_train)
one_hot_train_labels = file_reader.label_to_one_hot(training_labels)

test_data = file_reader.normalize_data(data_test)
test_labels = file_reader.normalize_labels(data_test)
one_hot_test_labels = file_reader.label_to_one_hot(test_labels)

# print(one_hot_train_labels)
#
print('INITIALIZE NETWORK')
model = NeuralNetwork(training_data, one_hot_train_labels, 128, 0.5)

print('Number of Epochs: ')
epochs = int(input())

print('1.) Use built from scratch network')
print('2.) Use keras network')
option = int(input())

if option == 1:
    for epoch in range(epochs):
        print('===============================')
        print(epoch)
        print('===============================')
        for x,y in zip(model.in_nodes, model.out_nodes):
            model.feedforward(x)
            model.backpropagation(x, y, epoch)
            loss = error(model.h2_out_activation, y)


    for n in range(10):
        print('Current Value')
        index = randint(0, 1000)
        file_reader.open_image(test_data[index])
        print(data_test[index][0])
        value = np.reshape(test_data[index], (1, test_data.shape[1]))
        predicted_value = model.generate_result(value)
        print('Predicted Value')
        print(predicted_value)

elif option == 2:
    model = neural_network_keras.generate_model(training_data, one_hot_train_labels)
    neural_network_keras.view_summary(model)
    neural_network_keras.train_model(model, training_data, one_hot_train_labels)

    scores = model.evaluate(test_data, one_hot_test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    for n in range(10):
        print('Current Value')
        index = randint(0, 1000)
        file_reader.open_image(test_data[index])
        print(data_test[index][0])
        value = np.reshape(test_data[index], (1, test_data.shape[1]))
        predicted_value = model.predict(value)
        print('Predicted Value')
        print(np.argmax(predicted_value))
