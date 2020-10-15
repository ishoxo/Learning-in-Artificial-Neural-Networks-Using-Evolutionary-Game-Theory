import numpy as np
import itertools


class mutation_neuron:
    """
    Individual reward neuron: generates probability distribution
    over possible strategies given state size
    """
    def __init__(self, input_size):
        self.input_size = input_size
        self.possible_states = self.generate_possible_states()
        self.strategies = self.generate_deterministic_strategies()[0]
        self.num_strategies = self.generate_deterministic_strategies()[1]
        self.mixed_strategy = self.initialize_mixed_strategy()
        self.fitness_metric = self.initialise_fitnesses()

    def initialise_fitnesses(self):
        n = self.num_strategies
        initial_fitnesses = [0] * n
        return initial_fitnesses

    def generate_possible_states(self):
        possible_states = list(itertools.product([0, 1], repeat=self.input_size))
        #possible_states = [list(tup) for tup in possible_states]
        return possible_states, len(possible_states)

    def generate_deterministic_strategies(self):
        deterministic_strats = list(itertools.product([0, 1], repeat=self.possible_states[1]))
        return deterministic_strats, len(deterministic_strats)

    def initialize_mixed_strategy(self):
        mixed_strategy = [1/self.num_strategies] * self.num_strategies
        return mixed_strategy

    def get_activation(self, state):
        #print('nms', self.mixed_strategy)
        chosen_strategy = np.random.choice(np.arange(0, self.num_strategies), p=self.mixed_strategy)
        #k = np.argwhere(self.possible_states[0] == np.array(state))
        k = self.possible_states[0].index(state)
        return self.strategies[chosen_strategy][k], chosen_strategy, k



class mutation_layer:
    """
    single layer for IR network: takes number of neurons and input size to each neuron as parameters, outputs binary vector.
    connections are the state size of the neurons
    """
    def __init__(self, num_neurons, connections, input_size):
        self.state_size = connections
        self.num_neurons = num_neurons
        self.neuron_list = self.initialise_neurons()
        self.input_size = input_size
        self.connection_map = self.show_connections()


    def show_connections(self):
        list = []
        for i in range(self.input_size):
            list.append(i)
        print('Input: ', list)
        print('Connections: ', self.split_state(list, self.input_size, self.state_size, self.num_neurons), 'Input size: ', self.input_size)
        return self.split_state(list, self.input_size, self.state_size, self.num_neurons)
    """
    output of previous layer is split into n parts, where n is the number of 
    neurons in the current layer. One neuron observes only one of the parts"
    """
    @staticmethod
    def split_state(state, input_size, connections, num_neurons):
        states = []
        length = input_size

        for i in range(num_neurons):
            s = []
            for j in range(connections):
                input = (i * (connections)) + j
                s.append(state[input % length])
            states.append(s)
        return states

    def initialise_neurons(self):
        neuron_list = []
        for i in range(self.num_neurons):
            neuron_list.append(mutation_neuron(input_size=self.state_size))
        return neuron_list

    def forward(self, state):
        states = self.split_state(state, self.input_size, self.state_size, self.num_neurons)
        output = []
        strategies = []
        state_indices = []
        for i in range(self.num_neurons):
            neuron = self.neuron_list[i]
            activation, strategy, state_index = neuron.get_activation(tuple(states[i]))
            output.append(activation)
            strategies.append(strategy)
            state_indices.append(state_index)
        return output, strategies, state_indices



class mutation_networks:
    """
    Individual reward network: composed of individual reward neurons in hierarchical structure
    """
    def __init__(self, neurons_in_each_layer, connections_in_each_layer, input_size):
        self.num_layers = len(neurons_in_each_layer)
        self.neurons_in_each_layer = neurons_in_each_layer
        self.connections_in_each_layer = connections_in_each_layer
        self.input_size = input_size
        self.topology = self.define_topology()
        self.network = self.initialise_network()
        self.neuron_list = self.list_neurons()


    def define_topology(self):
        print('Network, topology: ', self.input_size + self.neurons_in_each_layer)
        return self.input_size + self.neurons_in_each_layer


    def initialise_network(self):
        Layers = []
        for i in range(self.num_layers):
            layer = mutation_layer(num_neurons=self.neurons_in_each_layer[i],
                                   connections=self.connections_in_each_layer[i],
                                   input_size=self.topology[i])
            Layers.append(layer)
        return Layers

    def list_neurons(self):
        neuron_list = []
        for item in self.network:
            neuron_list.append(item.neuron_list)
        return neuron_list


    def forward(self, state):
        strategies = []
        state_indices = []
        for i in range(self.num_layers):
            layer = self.network[i]
            state, layer_strategies, layer_state_indices = layer.forward(state)
            strategies.append(layer_strategies)
            state_indices.append(layer_state_indices)
        return state, strategies, state_indices

    def save_mixed_strategy(self):
        mixed_strategies = []
        for i in range(self.num_layers):
            layer_strat = []
            for j in range(self.neurons_in_each_layer[i]):
                neuron = self.neuron_list[i][j]
                layer_strat.append(neuron.mixed_strategy)
            mixed_strategies.append(layer_strat)
        return mixed_strategies

    def load_mixed_strategy(self, strategy):
        for i in range(self.num_layers):
            for j in range(self.neurons_in_each_layer[i]):
                neuron = self.neuron_list[i][j]
                neuron.mixed_strategy = strategy[i][j]




example_network = mutation_networks([5, 4, 3, 2, 1], [3, 3, 3, 3, 2], [2])

neuron = example_network.neuron_list[0][0]
print(neuron)
x = neuron.mixed_strategy()

#print(neuron.mixed_strategy())



