import numpy as np
from ant_colony_optimization import AntColonyOptimization


class RunAco:
    def __init__(self, problem_name, file_path, num_ants, evaporation_rate, num_nodes, num_iter):
        """
        The purpose for this class is to abstract away the problem, and make the ACO implementation generic enough
        to handle any problem once the inputs are deconstructed from files.
        """
        self.problem_name = problem_name
        self.run(file_path, num_ants, evaporation_rate, num_nodes, num_iter)

    # create functions to read in the files and get the parameters needed to run ACO.
    def run(self, file_path, num_ants, evaporation_rate, num_nodes, num_iter):
        weight_matrix = np.loadtxt(file_path)
        adj_matrix = np.ones(weight_matrix.shape)  # only works for fully connected graphs
        for i in range(len(adj_matrix)):
            adj_matrix[i][i] = 0.0
        aco = AntColonyOptimization(num_ants, evaporation_rate, num_nodes, adj_matrix, weight_matrix, num_iter)
        aco.run()


def main():
    problem_name = 'Traveling Salesman Problem'
    # Link to dataset: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
    file_path = '../../data/48_cities.txt'
    num_ants = 48
    evaporation_rate = 0.25
    num_nodes = 48
    num_iter = 1000
    runAco = RunAco(problem_name, file_path, num_ants, evaporation_rate, num_nodes, num_iter)


if __name__ == '__main__':
    data = np.loadtxt("../../data/48_cities.txt")
    main()

