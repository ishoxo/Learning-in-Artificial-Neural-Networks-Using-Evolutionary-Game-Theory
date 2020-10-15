from mutation_network import mutation_networks
from matplotlib import pyplot as plt
from network_evaluation_functions import evaluate_network2, get_entropy, get_entropy_per_layer, graph_connections
from matplotlib import pyplot as plt
from torch.utils.data import Dataset as Dataset
import numpy as np
a = (0,0)
b = (1,0)
c = (0,1)
d = (1,1)
X = []
Y = []
for i in range(2500):
    X.append(a)
    X.append(b)
    X.append(c)
    X.append(d)
for i in range(2500):
    Y.append(0)
    Y.append(1)
    Y.append(1)
    Y.append(0)

x_val = X[:100]
y_val = Y[:100]


class toy_set(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        X_p = self.X[item]
        y_p = self.y[item]
        return X_p, y_p


train_dataset = toy_set(X, Y)
val_dataset = toy_set(x_val, y_val)
mn = mutation_networks([1], [2], [2])

learning_rate = 0.01
def train_toy_solver(network, num_epochs, train_dset, val_dset):
    entropies = []
    iterations = []
    layer_entropies = []
    validation_scores = []
    training_scores = []
    for epoch in range(num_epochs):
        epoch_accuracy = 0
        #entropies.append(get_entropy(network))
        #layer_entropies.append(get_entropy_per_layer(network))
        #validation_scores.append(evaluate_network2(network, val_dataset))
        print('Epoch: ', epoch)
        #print('Entropy: ', entropies[-1])
        #print('Validation Score: ', validation_scores[-1])
        for id, data in enumerate(train_dset):
            if id % 25 == 0:
                validation_scores.append(evaluate_network2(network, val_dataset))
                entropies.append(get_entropy(network))
                iterations.append(id)


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
        training_scores.append(epoch_accuracy / len(train_dset))
        print('Training Scores: ', training_scores[-1])

    plt.plot(iterations, entropies)
    plt.xlabel("Iteration")
    plt.ylabel("Neuron Strategy Entropy")
    plt.title(" LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    plt.plot(training_scores)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title(" LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()

    plt.plot(iterations, validation_scores)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.title(" LR:" + str(learning_rate))
    plt.grid(True)
    plt.show()



train_toy_solver(mn, 1, train_dataset, val_dataset)




