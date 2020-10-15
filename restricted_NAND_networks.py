import numpy as np
import itertools

"""
mutation nueron for NAND and mutation layer for NAND are normal; they are only used in the final layer,
where the state size is small enough to allow a neuron with unrestricted strategies
"""
class mutation_neuron_for_NAND:
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
        chosen_strategy = np.random.choice(np.arange(0, self.num_strategies), p=self.mixed_strategy)
        #k = np.argwhere(self.possible_states[0] == np.array(state))
        k = self.possible_states[0].index(tuple(state))
        return self.strategies[chosen_strategy][k], chosen_strategy

class mutation_layer_for_NAND:
    """
    single layer for IR network: takes number of neurons and input size to each neuron as parameters, outputs binary vector.
    connections are the state size of the neurons
    """
    def __init__(self, num_neurons, input_size):
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.neuron_list = self.initialise_neurons()


    def initialise_neurons(self):
        neuron_list = []
        for i in range(self.num_neurons):
            neuron_list.append(mutation_neuron_for_NAND(input_size=self.input_size))
        return neuron_list

    def forward(self, state):
        output = []
        strategies = []
        for i in range(self.num_neurons):
            neuron = self.neuron_list[i]
            activation, strategy = neuron.get_activation(state)
            output.append(activation)
            strategies.append(strategy)

        return output, strategies

class restricted_NAND_neuron:
    """
    mutation neuron with a small subset of strategies; only activates if a certain input is activated
    """

    def __init__(self, input_size):
        self.input_size = input_size
        self.strategies = self.generate_deterministic_strategies()[0]
        self.num_strategies = self.generate_deterministic_strategies()[1] + 1
        self.mixed_strategy = self.initialize_mixed_strategy()
        self.fitness_metric = self.initialise_fitnesses()

    @staticmethod
    def xor(input):
        if sum(input) < 2:
            return 1
        else:
            return 0

    def generate_deterministic_strategies(self):
        potential_connections = []
        for i in range(self.input_size):
            potential_connections.append(i)
        strategies = list(itertools.combinations(potential_connections, 2))
        #print(strategies)
        return strategies, len(strategies)

    def initialize_mixed_strategy(self):
        mixed_strategy = [1 / (self.num_strategies)] * (self.num_strategies)
        return mixed_strategy

    def get_activation(self, state):
        strategy_number = np.random.choice(np.arange(0, self.num_strategies), p=self.mixed_strategy)
        if strategy_number == self.num_strategies - 1:
            return 0, strategy_number
        else:
            chosen_strategy = self.strategies[strategy_number]
            inputs = [state[x] for x in chosen_strategy]
            return self.xor(inputs), strategy_number


mn = restricted_NAND_neuron(5)


class restricted_NAND_layer:
    """
    single layer for IR network: takes number of neurons and input size to each neuron as parameters, outputs binary vector.
    input size represents the number of connections each neuron has to the previous layer
    """

    def __init__(self, num_neurons, input_size):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.neuron_list = self.initialise_neurons()

    def initialise_neurons(self):
        neuron_list = []
        for i in range(self.num_neurons):
            neuron_list.append(restricted_NAND_neuron(input_size=self.input_size))
        return neuron_list

    def forward(self, state):
        # states = self.split_state(state, self.input_size, self.state_size, self.num_neurons)
        output = []
        strategies = []
        for i in range(self.num_neurons):
            neuron = self.neuron_list[i]
            activation, strategy = neuron.get_activation(state)
            output.append(activation)
            strategies.append(strategy)
        return output, strategies




class restricted_NAND_network:
    """
    Individual reward network: composed of individual reward neurons in hierarchical structure
    """

    def __init__(self, neurons_in_each_layer, input_size):
        self.num_layers = len(neurons_in_each_layer)
        self.neurons_in_each_layer = neurons_in_each_layer
        self.input_size = input_size
        self.topology = self.define_topology()
        self.network = self.initialise_network()
        self.neuron_list = self.list_neurons()

    def define_topology(self):
        #print('Network, topology: ', self.input_size + self.neurons_in_each_layer)
        return self.input_size + self.neurons_in_each_layer

    def initialise_network(self):
        Layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                layer = mutation_layer_for_NAND(num_neurons=1,
                                                input_size=self.topology[i])
            else:
                layer = restricted_NAND_layer(num_neurons=self.neurons_in_each_layer[i],
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
        for i in range(self.num_layers):
            layer = self.network[i]
            state, layer_strategies = layer.forward(state)
            strategies.append(layer_strategies)
        return state, strategies, 1

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
