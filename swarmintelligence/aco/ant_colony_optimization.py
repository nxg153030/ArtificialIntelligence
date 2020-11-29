import numpy as np
import random
from .ant import Ant
from .graph import ACOGraph


class AntColonyOptimization:
    def __init__(self, num_ants, evaporation_rate, num_iter):
        """
        Link to paper: http://users.dimi.uniud.it/~antonio.dangelo/Robotica/2018/helper/10.1.1.26.1865.pdf
        """
        self.num_ants = num_ants
        self.num_iter = num_iter
        self.evaporation_rate = evaporation_rate
        self.alpha = 0.0
        self.beta = 0.0
        self.init_trail_value = 0.0
        self.ant_colony = [Ant(i) for i in range(self.num_ants)]
        self.graph = ACOGraph(num_nodes, adj_matrix)

    def init_ant_colony(self):
        # Put a little bit of pheromone on each edge
        for i in range(self.graph.num_nodes):
            for j in range(self.graph.num_nodes):
                self.graph.pheromone_matrix[i][j] = self.init_trail_value

        # place N ants on M nodes
        for i in range(self.num_ants):
            self.ant_colony[i].current_node = random.randint(0, self.graph.num_nodes - 1)

    def run(self):
        self.init_ant_colony()
