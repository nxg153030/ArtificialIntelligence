import numpy as np
import random
from ant import Ant
from graph import ACOGraph

Q = 1.0


class AntColonyOptimization:
    def __init__(self, num_ants, evaporation_rate, num_nodes, adj_matrix, weight_matrix, num_iter):
        """
        Link to paper: http://users.dimi.uniud.it/~antonio.dangelo/Robotica/2018/helper/10.1.1.26.1865.pdf
        """
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.evaporation_rate = evaporation_rate
        self.alpha = 0.0  # relative importance of the trail
        self.beta = 0.0  # relative importance of visibility
        self.init_trail_value = 0.0
        self.ant_colony = [Ant(i, num_nodes) for i in range(self.num_ants)]
        self.graph = ACOGraph(num_nodes, adj_matrix, weight_matrix)

    def init_ant_colony(self):
        # Put a little bit of pheromone on each edge
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                self.graph.pheromone_matrix[i][j] = self.init_trail_value

        # place N ants on M nodes
        for i in range(self.num_ants):
            self.ant_colony[i].current_node = random.randint(0, self.graph.num_nodes - 1)
            self.ant_colony[i].tabu_list.append(self.ant_colony[i].current_node)  # place start city in tabu list

    def select_next_node(self, ant_idx):
        """select next node to visit for ant i"""
        current_node = self.ant_colony[ant_idx].current_node
        probability_list = []  # list containing probabilities for candidate cities to visit next

        # if there's only 1 unvisited city, return that city
        if len(self.ant_colony[ant_idx].available_nodes) == 1:
            next_node = self.ant_colony[ant_idx].available_nodes[0]
            self.ant_colony[ant_idx].available_nodes.pop()
            return next_node

        # for fully connected graphs, need to iterate through every edge.
        # No, not needed, only edges which are connected to the current city.
        for i in range(self.graph.num_nodes):
            if i != ant_idx:
                visibility = 1 / self.graph.weight_matrix[current_node][i]
                numerator = (self.graph.pheromone_matrix[current_node][i] ** self.alpha) * (visibility ** self.beta)
                denominator = 0.0
                for k in range(self.graph.num_nodes):
                    if k not in self.ant_colony[ant_idx].tabu_list and k != i:
                        denominator += (self.graph.pheromone_matrix[current_node][k] ** self.alpha) * \
                                       ((1/self.graph.weight_matrix[current_node][k]) ** self.beta)
                if denominator != 0.0:
                    transition_probability = numerator / denominator
                else:
                    transition_probability = 0.0
                probability_list.append((i, transition_probability))

        next_node = self.roulette_wheel_selection(probability_list)
        return next_node

    # https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
    def roulette_wheel_selection(self, probabilities):
        """
        TODO: Optimize to run in O(logn) or O(1)
        Further reading: https://www.keithschwarz.com/darts-dice-coins/
        """
        _max = sum(probabilities)
        pick = random.uniform(0, _max)
        current = 0.0
        for prob, idx in probabilities:
            current += prob
            if current >= pick:
                return prob

    def construct_ant_solutions(self):
        for i in range(self.num_ants):
            self.select_next_node(i)  # choose city to go to, using transition probability

        # drop pheromone on the edges

            pass

    def update_pheromones(self):
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                self.graph.pheromone_matrix[i][j] = (1 - self.evaporation_rate) * self.graph.pheromone_matrix[i][j] \
                                                    + self.graph.delta_pheromone[i][j]

    def clear_lists(self):
        pass

    def run(self):
        num_iter = 0
        self.init_ant_colony()
        while num_iter != self.num_iter:
            self.construct_ant_solutions()
            self.update_pheromones()
            self.clear_lists()
            num_iter += 1