import numpy as np


class ACOGraph:
    def __init__(self, num_nodes, adj_matrix, weight_matrix):
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix
        self.weight_matrix = weight_matrix
        self.pheromone_matrix = np.array((self.num_nodes, self.num_nodes))
        self.delta_pheromone_matrix = np.array((self.num_nodes, self.num_nodes))

