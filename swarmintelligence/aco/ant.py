import numpy as np


class Ant:
    def __init__(self, ant_idx, num_nodes):
        self.idx = ant_idx
        self.current_node = -1
        self.pheromone_matrix = np.zeros((num_nodes, num_nodes))
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        self.available_nodes = [i for i in range(num_nodes)]
        self.tour_length = 0
        self.tabu_list = [ant_idx]  # list of cities already visited
