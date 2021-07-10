import numpy as np
from ant_colony_optimization import AntColonyOptimization


class RunAco:
    def __init__(self, problem_name, file_path, num_ants, evaporation_rate, alpha, beta, Q, num_nodes, num_iter):
        """
        The purpose for this class is to abstract away the problem, and make the ACO implementation generic enough
        to handle any problem once the inputs are deconstructed from files.
        """
        self.problem_name = problem_name
        self.run(file_path, num_ants, evaporation_rate, Q, alpha, beta, num_nodes, num_iter)

    # create functions to read in the files and get the parameters needed to run ACO.
    def run(self, file_path, num_ants, evaporation_rate, Q, alpha, beta, num_nodes, num_iter):
        weight_matrix = np.loadtxt(file_path)
        adj_matrix = np.ones(weight_matrix.shape)  # only works for fully connected graphs
        for i in range(len(adj_matrix)):
            adj_matrix[i][i] = 0.0
        aco = AntColonyOptimization(num_ants, evaporation_rate, alpha, beta, Q, num_nodes, adj_matrix, weight_matrix,
                                    num_iter)
        aco.run()


def main():
    problem_name = 'Traveling Salesman Problem'
    # Link to dataset: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
    file_path = '../../data/15_cities.txt'
    num_ants = 15
    evaporation_rate = 0.01
    alpha = 1
    beta = 5
    Q = 100.0
    num_nodes = 15
    num_iter = 1000
    RunAco(problem_name, file_path, num_ants, evaporation_rate, alpha, beta, Q, num_nodes, num_iter)


if __name__ == '__main__':
    main()

