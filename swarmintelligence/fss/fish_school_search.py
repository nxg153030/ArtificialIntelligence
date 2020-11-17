import random
import numpy as np
from common.optimization_functions import rastrigin

upper_limit = 5.12
lower_limit = 2.5
global_optimum = 0.0
step_ind_init = 0.01
step_ind_final = 0.0001
step_ind = step_ind_init
step_vol = 2 * step_ind
random.seed(10)


class Fish:
    def __init__(self, _id, dimensions):
        self.id = _id
        self.position_vec = np.array([random.uniform(lower_limit, upper_limit) for _ in range(dimensions)])
        self.fitness = np.inf
        self.delta_fitness = 0.0
        self.displacement = 0.0
        self.weight = 1.0

    def set_position_vec(self, l: []):
        self.position_vec = l

    def set_weight(self, l: []):
        self.weight = l

    def update_weight(self):
        pass

    def update_position_vec(self):
        pass


class FishSchoolSearch:
    def __init__(self, num_fish, dimensions: int, fitness_func, num_iter: int):
        self.num_fish = num_fish
        self.dimensions = dimensions
        self.fish_school = [None] * self.num_fish
        self.fitness_func = fitness_func
        self.num_iter = num_iter
        self.fish_school_weight = 0.0
        self.delta_fitness_all = [0.0] * self.num_fish
        self.delta_position_all = [0.0] * self.num_fish
        self.max_delta_fitness = -np.inf
        self.populate_fish_school()
        self.barycenter = np.array([0.0] * self.dimensions)

    def populate_fish_school(self):
        for i in range(self.num_fish):
            self.fish_school[i] = Fish(i, self.dimensions)
            self.fish_school_weight += self.fish_school[i].weight

    def set_barycenter(self, fish_weights_sum):
        weighted_positions = np.array([fish.position_vec * fish.weight for fish in self.fish_school])
        weighted_position_sum = weighted_positions.sum(axis=0)
        self.barycenter = np.true_divide(weighted_position_sum, fish_weights_sum)

    def instinctive_movement(self):
        resulting_direction = sum([delta_pos * delta_fitness for delta_pos, delta_fitness in
                                   zip(self.delta_position_all, self.delta_fitness_all)])
        for fish in self.fish_school:
            fish.position_vec = fish.position_vec + resulting_direction

    def volitive_movement(self):
        fish_weights_current = sum(fish.weight for fish in self.fish_school)
        self.set_barycenter(fish_weights_current)
        for fish in self.fish_school:
            if fish_weights_current > self.fish_school_weight:
                fish.position_vec = fish.position_vec - (((step_vol * random.uniform(0, 1)) *
                                                          np.subtract(fish.position_vec, self.barycenter)) /
                                                         np.linalg.norm(fish.position_vec - self.barycenter))
            else:
                fish.position_vec = fish.position_vec + (((step_vol * random.uniform(0, 1)) *
                                                          np.subtract(fish.position_vec, self.barycenter)) /
                                                         np.linalg.norm(fish.position_vec - self.barycenter))
        self.fish_school_weight = fish_weights_current

    def stopping_condition_met(self, num_iter, fitness):
        return num_iter >= self.num_iter or fitness <= global_optimum

    def log_individual_fitness(self, iter_counter):
        print(f'Iteration: {iter_counter}')
        for fish in self.fish_school:
            print(f'Fish ID: {fish.id}, Fitness: {fish.fitness}')

    def run(self):
        """
        initialize all fish in random positions
        while stop criterion is not met
            for each fish:
                evaluate fitness function - done
                perform individual movement - done
                feeding - done (weight update)
                evaluate fitness function

            for each fish:
                perform instinctive movement

            calculate barycenter

            for each fish:
                perform volitive movement
            update step

        TODO:
        """
        iter_counter = 0
        fitness = 1000
        global step_ind, step_vol, step_ind_final
        while not self.stopping_condition_met(iter_counter, fitness):
            for idx, fish in enumerate(self.fish_school):
                current_fitness = self.fitness_func(fish.position_vec, self.dimensions)
                # individual movement
                new_candidate_pos = fish.position_vec + (random.uniform(-1, 1) * step_ind)  # eq 1
                new_fitness = self.fitness_func(new_candidate_pos, self.dimensions)  # eq 2
                delta_fitness = new_fitness - current_fitness
                if delta_fitness < 0:
                    fish.position_vec = new_candidate_pos
                else:
                    delta_fitness = 0  # if curr_fitness is worse, then don't move

                fish.fitness = new_fitness  # update the fitness for fish every iteration, even if it is 0
                fish.delta_fitness = abs(delta_fitness)
                self.delta_fitness_all[idx] = delta_fitness
                # update max_delta_fitness
                if fish.delta_fitness > self.max_delta_fitness:
                    self.max_delta_fitness = fish.delta_fitness

            # weight update for all fish
            for fish in self.fish_school:
                fish.weight = fish.weight + (
                        fish.delta_fitness / self.max_delta_fitness)

            self.instinctive_movement()
            self.volitive_movement()

            self.log_individual_fitness(iter_counter)
            # step update
            step_ind = step_ind - (step_ind_init - step_ind_final) / self.num_iter
            iter_counter += 1


def main():
    num_fish = 10
    dimensions = 2
    fitness_func = rastrigin
    num_iter = 50
    fss = FishSchoolSearch(num_fish, dimensions, fitness_func, num_iter)
    fss.run()


if __name__ == '__main__':
    main()
