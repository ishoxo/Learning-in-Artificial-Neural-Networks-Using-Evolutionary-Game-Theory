import numpy as np
import itertools
import pandas as pd
from matplotlib import pyplot as plt

class rps_neuron:
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
        deterministic_strats = list(itertools.product([0, 1, 2], repeat=self.possible_states[1]))
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


my_rps = rps_neuron(input_size=0)

rock_strat = [0.99, 0.005, 0.005]
paper_strat = [0.005, 0.99, 0.005]
rock_neuron = rps_neuron(input_size=0)
paper_neuron = rps_neuron(input_size=0)

rock_neuron.mixed_strategy = rock_strat
paper_neuron.mixed_strategy = paper_strat

# print('here', rock_neuron.possible_states[0])
#
# print('here1', rock_neuron.get_activation(())[0])
#
# print('here2', paper_neuron.get_activation(()))

payoffs = [[[0, 0], [-1, 1], [1, -1]], [[1, -1], [0, 0], [-1, 1]], [[-1, 1], [1, -1], [0, 0]]]




def play_game(neuron1, neuron2, learning_rate, num_games):
    neuron1_strategies = []
    neuron2_strategies = []
    for i in range(num_games):
        neuron1_strategies.append(neuron1.mixed_strategy)
        neuron2_strategies.append(neuron2.mixed_strategy)
        neuron1_action, neuron1_strat, _ = neuron1.get_activation(())
        neuron2_action, neuron2_strat, _ = neuron2.get_activation(())

        payoff = payoffs[neuron1_action][neuron2_action]
        fitness1 = payoff[0]
        fitness2 = payoff[1]

        neuron1.mixed_strategy = [max(0, (1 - (learning_rate * fitness1)) * item) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [max(0, (1 - (learning_rate * fitness2)) * item) for item in neuron2.mixed_strategy]

        neuron1.mixed_strategy[neuron1_strat] += learning_rate * fitness1
        neuron2.mixed_strategy[neuron2_strat] += learning_rate * fitness2

        neuron1.mixed_strategy = [max(0, (1 - (learning_rate * fitness1)) * item) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [max(0, (1 - (learning_rate * fitness2)) * item) for item in neuron2.mixed_strategy]

        neuron1.mixed_strategy = [min(1, item) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [min(1, item) for item in neuron2.mixed_strategy]

        neuron1.mixed_strategy = [item / sum(neuron1.mixed_strategy) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [item / sum(neuron2.mixed_strategy) for item in neuron2.mixed_strategy]
    return neuron1_strategies, neuron2_strategies




def play_game_mut(neuron1, neuron2, learning_rate, num_games, mutation_rate):
    neuron1_strategies = []
    neuron2_strategies = []
    for i in range(num_games):
        neuron1_strategies.append(neuron1.mixed_strategy)
        neuron2_strategies.append(neuron2.mixed_strategy)
        neuron1_action, neuron1_strat, _ = neuron1.get_activation(())
        neuron2_action, neuron2_strat, _ = neuron2.get_activation(())

        payoff = payoffs[neuron1_action][neuron2_action]
        fitness1 = payoff[0]
        fitness2 = payoff[1]

        neuron1.mixed_strategy = [max(0, (1 - (learning_rate * fitness1)) * item) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [max(0, (1 - (learning_rate * fitness2)) * item) for item in neuron2.mixed_strategy]

        neuron1.mixed_strategy[neuron1_strat] += learning_rate * fitness1
        neuron2.mixed_strategy[neuron2_strat] += learning_rate * fitness2

        neuron1.mixed_strategy = [max(0, (1 - (learning_rate * fitness1)) * item) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [max(0, (1 - (learning_rate * fitness2)) * item) for item in neuron2.mixed_strategy]

        neuron1.mixed_strategy = [min(1, item) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [min(1, item) for item in neuron2.mixed_strategy]

        if mutation_rate > 0:
            neuron1.mixed_strategy[0] = ((1 - mutation_rate) * neuron1.mixed_strategy[0]) + (
                    mutation_rate * neuron1.mixed_strategy[1])
            neuron1.mixed_strategy[1] = ((1 - mutation_rate) * neuron1.mixed_strategy[1]) + (
                    mutation_rate * neuron1.mixed_strategy[0])
            neuron2.mixed_strategy[0] = ((1 - mutation_rate) * neuron2.mixed_strategy[0]) + (
                    mutation_rate * neuron2.mixed_strategy[1])
            neuron2.mixed_strategy[1] = ((1 - mutation_rate) * neuron2.mixed_strategy[1]) + (
                    mutation_rate * neuron2.mixed_strategy[0])

        neuron1.mixed_strategy = [item / sum(neuron1.mixed_strategy) for item in neuron1.mixed_strategy]
        neuron2.mixed_strategy = [item / sum(neuron2.mixed_strategy) for item in neuron2.mixed_strategy]
    return neuron1_strategies, neuron2_strategies


columns = ['Rock', 'Paper', 'Scissors']
s1, s2 = play_game(rock_neuron, paper_neuron, 0.005, 5000)
#s1, s2 = play_game_mut(rock_neuron, paper_neuron, 0.005, 10000, 0.001)
print(s1[0], s1[-1])
print(s2[0], s2[-1])

layer_dataframe = pd.DataFrame(s1, columns= columns)
plt.figure()
layer_dataframe.plot()

# plt.plot(entropies)
plt.xlabel("Probability of Action")
plt.ylabel("Game No.")
plt.title("Player 1")
plt.grid(True)
plt.show()


layer_dataframe2 = pd.DataFrame(s2, columns=columns)
plt.figure()
layer_dataframe2.plot()

# plt.plot(entropies)
plt.xlabel("Probability of Action")
plt.ylabel("Game No.")
plt.title("Player 2")
plt.grid(True)
plt.show()

