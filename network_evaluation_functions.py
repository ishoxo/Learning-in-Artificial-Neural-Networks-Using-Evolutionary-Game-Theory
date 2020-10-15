from scipy.stats import entropy
from matplotlib import pyplot as plt
import numpy as np
import random
import time
import itertools
import copy
import networkx as nx
import pandas as pd


'''
Converts mixed strategies of all neurons in network into deterministic strategies: 
the deterministic strategy with the highest sampling probability is assigned a
sampling probability of 1, all other are assigned a probability of 0
'''
def convert_to_det(network):
    for i in range(network.num_layers):
        for j in range(network.neurons_in_each_layer[i]):
            neuron = network.neuron_list[i][j]
            k = int(np.argmax(neuron.mixed_strategy))
            neuron.mixed_strategy = [x * 0 for x in neuron.mixed_strategy]
            neuron.mixed_strategy[k] = 1
    return network

'''
Evaluates accuracy of network on dataset
'''
def evaluate_network2(network, dataset):
    score = 0
    for id, data in enumerate(dataset):
        X, y = data
        output, strategies, _ = network.forward(X)
        output = int(output[0])
        if output == y:
            score += 1

    score = score / len(dataset)
    return score

def evaluate_network3(network, dataset):
    score = 0
    for id, data in enumerate(dataset):
        X, y = data
        output, strategies = network.forward(X)
        y = np.asarray(y)
        output = np.asarray(output)
        comparison = output == y
        if comparison.all():
            score += 1

    score = score / len(dataset)
    return score

"""
The entropy of the mixed strategy used by each neuron in the network is computed, the 
mean entropy is returned
"""
def get_entropy(network):
    entropy_values = []
    for i in range(len(network.neuron_list)):
        for j in range(len(network.neuron_list[i])):
            neuron = network.neuron_list[i][j]
            strategy = neuron.mixed_strategy
            entropy_values.append(entropy(strategy))
    return sum(entropy_values)/len(entropy_values)



'''
For each layer, the entropy of all neurons in the layer is computed and the mean 
entropy value for that layer is returned
'''
def get_entropy_per_layer(network):
    entropy_values = []
    for i in range(len(network.neuron_list)):
        layer_entropy = []
        for j in range(len(network.neuron_list[i])):
            neuron = network.neuron_list[i][j]
            strategy = neuron.mixed_strategy
            layer_entropy.append(entropy(strategy))
        entropy_values.append(sum(layer_entropy) / len(layer_entropy))
    return entropy_values

"""
Using for assigning credit during training.
For each neuron, the network is evaluated having replaced this neuron with
a random agent (activation chosen randomly).
"""
def difference_evaluation2(network, dataset):
    evaluations = []
    initial_score = evaluate_network2(network, dataset)
    for i in range(len(network.neuron_list)):
        layer_evaluations = []
        for j in range(len(network.neuron_list[i])):
            neuron = network.neuron_list[i][j]
            mixed_strategy = neuron.mixed_strategy
            neuron.mixed_strategy = neuron.initialize_mixed_strategy()
            score = evaluate_network2(network, dataset)
            layer_evaluations.append(score)
            neuron.mixed_strategy = mixed_strategy
        evaluations.append(layer_evaluations)
    credit = []
    differences = []
    for i in range(len(evaluations)):
        layer_differences = []
        for j in range(len(evaluations[i])):
            layer_differences.append(initial_score - evaluations[i][j])
        differences.append(layer_differences)
    credit = differences
    # for i in range(len(differences)):
    #     layer_credit = []
    #     for j in range(len(differences[i])):
    #
    #     layer_credit.append(differences[i] / sum(differences)

    return initial_score, evaluations , differences, credit



"""
Measures the strength of the connection between two neurons A and B,
where A is the feeds into B and whose activation is the c'th element of
B's observed state.
Returns the expected increase in the activation of B if A's activation 
increases from 0 to 1.
"""
def get_connection(neuron, c):
    """
    :param neuron: downstream neuron, that neuron 1 feeds into
    :param c: position of neuron 1 in state representation for neuron 2
    :return: connection between neuron 1
    """
    possible_states = neuron.possible_states
    deterministic_strats = neuron.strategies
    mixed_strat = neuron.mixed_strategy

    state_responses = [0] * len(possible_states[0])
    for i in range(len(deterministic_strats)):
        p = mixed_strat[i]
        single_strat = [x * p for x in deterministic_strats[i]]
        state_responses = [x + y for x, y in zip(state_responses, single_strat)]

    activation_without_neuron1_activating = 0
    activation_with_neuron1_activating = 0

    for j in range(len(possible_states[0])):

        if possible_states[0][j][c] == 1:

            activation_with_neuron1_activating += state_responses[j]
        else:
            activation_without_neuron1_activating += state_responses[j]
    strength = 2 * (activation_with_neuron1_activating - activation_without_neuron1_activating) / len(possible_states)

    return float("{:.2f}".format(strength))

"""
creates graph representing the network.
edges in the graph are labelled with connection strength given by function above
"""
def graph_connections(network):
    '''
    :param network: trained IR network
    :return: annotated graph showing strength of connections
    '''
    connections = []
    for i in range(network.num_layers):
        layer_list = []
    #     layer = network.network[i]
    #     layer_connections = layer.connection_map
    #     connections.append(layer_connections)
    # print('quick connect: ', connections)
        for j in range(network.neurons_in_each_layer[i]):
            neuron_connection = []
            neuron = network.neuron_list[i][j]
            for c in range(neuron.input_size):
                synapse_strength = get_connection(neuron, c)
                neuron_connection.append(synapse_strength)
            layer_list.append(neuron_connection)
        connections.append(layer_list)


    topology = network.topology
    pos = {}
    node_list = []
    for i in range(len(topology)):
        layer_list = []
        for j in range(topology[i]):
            name = 'L' + str(i) + 'N' + str(j)
            layer_list.append(name)
            pos[name] = ((i + 1) * (1 / (len(topology) + 1)), 1 - ((j + 1) * (1 / (topology[i] + 1))))
            node_list.append(layer_list)

    edges = []
    edge_labels = {}
    for i in range(network.num_layers):
        layer = network.network[i]
        connection_list = []
        for k in range(layer.input_size):
            connection_list.append(k)
        connection_map = layer.split_state(connection_list, len(connection_list), layer.state_size, layer.num_neurons)
        #print('Layer ' + str(i) + ' connections: ', connection_map)
        for j in range(network.neurons_in_each_layer[i]):
            n = i + 1
            name = 'L' + str(n) + 'N' + str(j)
            for t in range(len(connection_map[j])):
                upstream_neuron = connection_map[j][t]
                upstream_neuron_name = 'L' + str(i) + 'N' + str(upstream_neuron)
                new_edge = [name, upstream_neuron_name]
                edges.append(new_edge)
                strength = connections[i][j][t]
                edge_labels[(name, upstream_neuron_name)] = str(strength)



    G = nx.Graph()
    for i in range(len(node_list)):
        G.add_nodes_from(node_list[i])
    G.add_edges_from(edges)
    plt.figure()
    nx.draw(G, pos, edge_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    #
    # print(edges)
    # print(edge_labels)
    #
    # nx.draw(G, pos=pos, edge_labels=edge_labels, font_color= 'red', with_labels=True)
    plt.show()


