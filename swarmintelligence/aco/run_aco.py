from .ant_colony_optimization import AntColonyOptimization


class RunAco:
    def __init__(self, problem_name, file_path):
        """
        The purpose for this class is to abstract away the problem, and make the ACO implementation generic enough
        to handle any problem once the inputs are deconstructed from files.
        """
        self.problem_name = problem_name
        self.problem_file_path = file_path

    # create functions to read in the files and get the parameters needed to run ACO.
    def run(self):
        aco = AntColonyOptimization(num_ants, evaporation_rate, num_nodes, adj_matrix, weight_matrix, num_iter)
        aco.run()


if __name__ == '__main__':
    runAco = RunAco()
