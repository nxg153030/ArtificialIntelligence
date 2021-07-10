import copy
import numpy as np
import random
from ant import Ant
from graph import ACOGraph


class AntColonyOptimization:
    def __init__(self, num_ants, evaporation_rate, alpha, beta, Q, num_nodes, adj_matrix, weight_matrix, num_iter):
        """
        Link to paper: http://users.dimi.uniud.it/~antonio.dangelo/Robotica/2018/helper/10.1.1.26.1865.pdf
        """
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.alpha = alpha  # relative importance of the trail
        self.beta = beta  # relative importance of visibility
        self.init_trail_value = 0.0
        self.ant_colony = [Ant(i, num_nodes) for i in range(self.num_ants)]
        self.graph = ACOGraph(num_nodes, adj_matrix, weight_matrix)
        self.shortest_path = []
        self.shortest_path_length = np.inf

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
        probability_dict = dict()  # list containing probabilities for candidate cities to visit next

        # if there's only 1 unvisited city, return that city
        # TODO: Use a set for available_nodes
        if len(self.ant_colony[ant_idx].available_nodes) == 1:
            next_node = self.ant_colony[ant_idx].available_nodes[0]
            return next_node

        # for fully connected graphs, need to iterate through every edge.
        # No, not needed, only edges which are connected to the current city.
        for i in self.ant_colony[ant_idx].available_nodes:
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
                # probability_list.append((i, transition_probability))
                probability_dict[i] = transition_probability

        next_node = self.roulette_wheel_selection(probability_dict)
        return next_node

    # https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
    def roulette_wheel_selection(self, probabilities):
        """
        TODO: Optimize to run in O(logn) or O(1)
        Further reading: https://www.keithschwarz.com/darts-dice-coins/
        """
        _max = sum(probabilities.values())
        pick = random.uniform(0, _max)
        current = 0.0
        for city, prob in probabilities.items():
            current += prob
            if current >= pick:
                return city

    def update_ant_tour_details(self, ant_idx, selected_node):
        last_visited_city = self.ant_colony[ant_idx].tabu_list[-1]
        self.ant_colony[ant_idx].tabu_list.append(selected_node)
        self.ant_colony[ant_idx].tour_length += self.graph.weight_matrix[last_visited_city][selected_node]
        self.ant_colony[ant_idx].available_nodes.remove(selected_node)
        self.ant_colony[ant_idx].adjacency_matrix[last_visited_city][selected_node] = 1

    def construct_ant_solutions(self):
        for i in range(self.num_ants):
            # each ant completes a full tour of the search space
            while self.ant_colony[i].available_nodes:
                next_node = self.select_next_node(i)  # choose city to go to, using transition probability
                self.update_ant_tour_details(i, next_node)

        # compute tour lengths for each ant
        for i in range(self.num_ants):
            self.ant_colony[i].tabu_list.append(self.ant_colony[i].tabu_list[0])  # Add the starting city to complete the tour.
            last_visited_city, current_city = self.ant_colony[i].tabu_list[-2], self.ant_colony[i].tabu_list[-1]
            self.ant_colony[i].tour_length += self.graph.weight_matrix[last_visited_city][current_city]
            if self.ant_colony[i].tour_length < self.shortest_path_length:
                self.shortest_path_length = self.ant_colony[i].tour_length
                self.shortest_path = copy.deepcopy(self.ant_colony[i].tabu_list)

            # drop pheromone on the search space
            for j in range(self.graph.num_nodes):
                for k in range(self.graph.num_nodes):
                    if j != k:
                        if self.ant_colony[i].adjacency_matrix[j][k] == 1:
                            self.ant_colony[i].pheromone_matrix[j][k] = self.Q / self.ant_colony[i].tour_length  # calculate pheromone dropped by ant i on edge (j, k)
                            self.graph.delta_pheromone_matrix[j][k] += self.ant_colony[i].pheromone_matrix[j][k]  # add to the pheromone to edge (j, k) for current iteration
                        else:
                            self.ant_colony[i].pheromone_matrix[j][k] = 0

    def update_pheromones(self):
        # Update trail intensity
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                self.graph.pheromone_matrix[i][j] = (1 - self.evaporation_rate) * self.graph.pheromone_matrix[i][j] \
                                                    + self.graph.delta_pheromone_matrix[i][j]

    def clear_lists(self):
        for i in range(self.num_ants):
            self.ant_colony[i].tour_length = 0
            self.ant_colony[i].pheromone_matrix.fill(0.0)
            self.ant_colony[i].tabu_list.clear()
            self.ant_colony[i].adjacency_matrix.fill(0)
            self.ant_colony[i].available_nodes = copy.deepcopy(self.graph.node_list)
            self.ant_colony[i].tabu_list.append(i)  # all ants start at the same city, can add randomization
            self.ant_colony[i].current_node = i
            self.ant_colony[i].available_nodes.remove(self.ant_colony[i].tabu_list[-1])

    def run(self):
        num_iter = 0
        self.init_ant_colony()
        while num_iter != self.num_iter:
            self.construct_ant_solutions()
            self.update_pheromones()
            self.clear_lists()
            print(f'Shortest path after iteration {num_iter+1}: {self.shortest_path}')
            print(f'Shortest path length after first iteration {num_iter+1}: {self.shortest_path_length}')
            num_iter += 1
        print(f'Shortest path: {self.shortest_path}')
        print(f'Shortest path length: {self.shortest_path_length}')
