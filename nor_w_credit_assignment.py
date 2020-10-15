# -*- coding: utf-8 -*-
"""NOR_w_credit_assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ry_5EdPaT9KHAsrleFbfJk1Q6Te2zI63
"""

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'Individual_reward_networks/'
import os
import sys
sys.path.append(base_dir)
os.chdir(base_dir)

from network_evaluation_functions import evaluate_network2, get_entropy, get_entropy_per_layer, graph_connections, difference_evaluation2
from matplotlib import pyplot as plt
from all_versus_1_MNIST import transform_data2
import pandas as pd
import pickle
from torch.utils.data import Dataset as Dataset
from one_v_all_function import one_vs_all_data
from all_versus_1_MNIST import transform_data
from restricted_NOR_networks import restricted_NOR_network
import pickle

rn1 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn2 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn3 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn4 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn5 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn6 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn7 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn8 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn9 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
rn0 = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])

networks = [rn0, rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8, rn9]


train_size = 60000
MNIST_data = pd.read_csv('TSNE_all_untouched')
MNIST_train = MNIST_data[:train_size]
MNIST_test = MNIST_data[train_size:]

class MNIST_set(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        X_p = self.X[item]
        y_p = self.y[item]
        return X_p, y_p

# dataframe = one_vs_all_data(MNIST_train, 9)
# dataframe_val = one_vs_all_data(MNIST_test, 9)
# X_train, y_train = transform_data(dataframe)
# X_val, y_val = transform_data(dataframe_val)

# train_dataset = MNIST_set(X_train, y_train)
# val_dataset = MNIST_set(X_val, y_val)


def train_model_with_DE(my_network, num_epochs, learning_rate, dataset, val_dataset):
    print('Starting Training:')
    entropies = []
    layer_entropies = []
    validation_scores = []
    training_scores =[]
    for epoch in range(num_epochs):
        print('Starting Epoch:', epoch)
        if epoch % 5 == 0:
            initial_score, evaluations, differences, credit = difference_evaluation2(my_network, dataset)
            print(evaluations)
        layer_entropies.append(get_entropy_per_layer(my_network))
        entropies.append(get_entropy(my_network))
        print('Entropy: ', entropies[-1])
        validation_score = evaluate_network2(my_network, val_dataset)
        print('Validation Score: ', validation_score)
        validation_scores.append(validation_score)
        epoch_accuracy = 0
        for id, data in enumerate(dataset):
            X, y = data
            output, strategies, _ = my_network.forward(X)
            output = int(output[0])
            error = (y - output) ** 2
            if error == 0:
                epoch_accuracy += 1
            fitness = 1 - error
            for i in range(len(my_network.neuron_list)):
                for j in range(len(my_network.neuron_list[i])):
                    neuron = my_network.neuron_list[i][j]
                    chosen_strategy = strategies[i][j]
                    neuron_fitness = fitness - evaluations[i][j]
                    #neuron_fitness = fitness * credit[i][j]
                    neuron.mixed_strategy = [max(0, (1 - (learning_rate * neuron_fitness)) * item) for item in neuron.mixed_strategy]
                    neuron.mixed_strategy[chosen_strategy] += (max(0, learning_rate * neuron_fitness))
                    neuron.mixed_strategy = [item / sum(neuron.mixed_strategy) for item in neuron.mixed_strategy]
        training_scores.append(epoch_accuracy/len(dataset))
        print('Training Score:', training_scores[-1])
    plt.plot(entropies)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Entropy")
    plt.title(" LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    columns = []
    for i in range(my_network.num_layers):
        j = i + 1
        columns.append('Layer ' + str(j))
    layer_dataframe = pd.DataFrame(layer_entropies, columns=columns)
    plt.figure()
    layer_dataframe.plot()

    # plt.plot(entropies)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Entropy")
    plt.title(" LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    plt.plot(validation_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(" LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    return entropies, training_scores, validation_scores, my_network.save_mixed_strategy()

with open('NOR_wca_strat_65.pkl', 'rb') as f:
    strats = pickle.load(f)

classifier_strategies = []
classifier_layer_entropies = []
classifier_train_scores = []
classifier_validation_scores = []
for i in range(10):
    #NOR_network = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
    NOR_network = networks[i]
    NOR_network.load_mixed_strategy(strats[i])
    dataframe = one_vs_all_data(MNIST_train, i)
    dataframe_val = one_vs_all_data(MNIST_test, i)
    X_train, y_train = transform_data(dataframe)
    X_val, y_val = transform_data(dataframe_val)
    train_dataset = MNIST_set(X_train, y_train)
    val_dataset = MNIST_set(X_val, y_val)
    entropies, train_score, val_score, strategy = train_model_with_DE(NOR_network, 15, 0.005, train_dataset, val_dataset)
    classifier_strategies.append(strategy)
    classifier_layer_entropies.append(entropies)
    classifier_train_scores.append(train_score)
    classifier_validation_scores.append(val_score)


import pickle
with open('NOR_wca_train_80.pkl', 'wb') as f:
    pickle.dump(classifier_train_scores, f)
    
with open('NOR_wca_val_80.pkl', 'wb') as f:
    pickle.dump(classifier_validation_scores, f)
    
with open('NOR_wca_strat_80.pkl', 'wb') as f:
    pickle.dump(classifier_strategies, f)
    
with open('NOR_wca_ent_80.pkl', 'wb') as f:
    pickle.dump(classifier_layer_entropies, f)
