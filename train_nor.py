# -*- coding: utf-8 -*-
"""TRAIN_NOR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eegNt_rSJL0nQsOl7M0yYhgvaNJTPpPi
"""

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'Individual_reward_networks/'
import os
import sys
sys.path.append(base_dir)
os.chdir(base_dir)

from network_evaluation_functions import evaluate_network2, get_entropy, get_entropy_per_layer, graph_connections
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from mutation_network import mutation_networks
from restricted_networks import restricted_network
from torch.utils.data import Dataset as Dataset
from one_v_all_function import one_vs_all_data
from all_versus_1_MNIST import transform_data
import numpy as np
from scipy.stats import entropy
from restricted_XOR_networks import restricted_XOR_network
from restricted_NOR_networks import restricted_NOR_network

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


def train_model_restricted(network, learning_rate, num_epochs, mutation_rate, dataset, val_dataset):
    entropies = []
    layer_entropies = []
    validation_scores = []
    training_scores = []
    for epoch in range(num_epochs):
        epoch_accuracy = 0
        entropies.append(get_entropy(network))
        layer_entropies.append(get_entropy_per_layer(network))
        validation_scores.append(evaluate_network2(network, val_dataset))
        print('Epoch: ', epoch)
        print('Entropy: ', entropies[-1])
        print('Validation Score: ', validation_scores[-1])
        if entropies[-1] < 0.05:
            break
        for id, data in enumerate(dataset):
            # if id % 1000 == 0:
            #     print(id/len(dataset))
            X, y = data
            y = np.asarray(y)
            output, strategies, _ = network.forward(X)

            error = (y - output) ** 2
            error = error[0]
            if error == 0:
                epoch_accuracy += 1
            fitness = 1 - error

            "update according to fitness"
            if fitness != 0:
                update = (1 - (learning_rate * fitness))
                for i in range(network.num_layers):
                    for j in range(network.neurons_in_each_layer[i]):
                        neuron = network.neuron_list[i][j]
                        chosen_strategy = strategies[i][j]
                        neuron.mixed_strategy = [update * item for item in neuron.mixed_strategy]
                        neuron.mixed_strategy[chosen_strategy] += (learning_rate * fitness)
                        neuron.mixed_strategy = [item / sum(neuron.mixed_strategy) for item in neuron.mixed_strategy]
            "update according to mutation"
            if mutation_rate != 0:
                for i in range(network.num_layers):
                    for j in range(network.neurons_in_each_layer[i]):
                        neuron = network.neuron_list[i][j]
                        n = neuron.num_strategies
                        neuron.mixed_strategy = [((1 - mutation_rate) * item) + ((1-item) * mutation_rate/n) for item in
                                                 neuron.mixed_strategy]
                        neuron.mixed_strategy = [item / sum(neuron.mixed_strategy) for item in neuron.mixed_strategy]
        training_scores.append(epoch_accuracy / len(dataset))
        print('Training Scores: ', training_scores[-1])

    columns = []
    for i in range(network.num_layers):
        j = i+1
        columns.append('Layer ' + str(j))
    layer_dataframe = pd.DataFrame(layer_entropies, columns=columns)
    plt.figure()
    layer_dataframe.plot()

    plt.xlabel("Epoch")
    plt.ylabel("Mean Layer Entropy")
    plt.title("MR: " + str(mutation_rate) + " LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    plt.plot(training_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("MR: " + str(mutation_rate) + " LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    plt.plot(validation_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("MR: " + str(mutation_rate) + " LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    return network.save_mixed_strategy(), layer_dataframe, validation_scores, training_scores

with open('NOR_strat_40+40.pkl', 'rb') as f:
    strats = pickle.load(f)

# classifier_strategies = []
# classifier_layer_entropies = []
# classifier_train_scores = []
# classifier_validation_scores = []
# for i in range(10):
#     NOR_network = restricted_NOR_network([9, 9, 9, 6, 3, 2, 1], [9])
#     NOR_network.load_mixed_strategy(strats[i])
#     dataframe = one_vs_all_data(MNIST_train, i)
#     dataframe_val = one_vs_all_data(MNIST_test, i)
#     X_train, y_train = transform_data(dataframe)
#     X_val, y_val = transform_data(dataframe_val)
#     train_dataset = MNIST_set(X_train, y_train)
#     val_dataset = MNIST_set(X_val, y_val)
#     strat, entropies, val_score, train_score = train_model_restricted(NOR_network, 0.001, 10, 0, train_dataset, val_dataset)
#     classifier_strategies.append(strat)
#     classifier_layer_entropies.append(entropies)
#     classifier_train_scores.append(train_score)
#     classifier_validation_scores.append(val_score)

NOR_network = restricted_NOR_network([9, 9, 9, 3, 2, 1], [9])
dataframe = one_vs_all_data(MNIST_train, 9)
dataframe_val = one_vs_all_data(MNIST_test, 9)
X_train, y_train = transform_data(dataframe)
X_val, y_val = transform_data(dataframe_val)
train_dataset = MNIST_set(X_train, y_train)
val_dataset = MNIST_set(X_val, y_val)
strat, entropies, val_score, train_score = train_model_restricted(NOR_network, 0.001, 10, 0, train_dataset, val_dataset)



# with open('NOR_train_40+50.pkl', 'wb') as f:
#     pickle.dump(classifier_train_scores, f)
    
# with open('NOR_val_40+50.pkl', 'wb') as f:
#     pickle.dump(classifier_validation_scores, f)
    
# with open('NOR_strat_40+50.pkl', 'wb') as f:
#     pickle.dump(classifier_strategies, f)
    
# with open('NOR_entropies_40+50.pkl', 'wb') as f:
#     pickle.dump(classifier_layer_entropies, f)

strat1, entropies1, val_score1, train_score1 = train_model_restricted(NOR_network, 0.001, 30, 0, train_dataset, val_dataset)

strat2, entropies2, val_score2, train_score2 = train_model_restricted(NOR_network, 0.001, 30, 0, train_dataset, val_dataset)