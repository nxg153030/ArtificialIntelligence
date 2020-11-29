class Ant:
    def __init__(self, ant_idx):
        self.idx = ant_idx
        self.current_node = -1
        self.tabu_list = []  # list of cities already visited
