from swarmintelligence.fss import FishSchoolSearch


class NeuralNetFss:
    def __init__(self, layers, learning_rate, error_func):
        self.layers = layers
        self.weights = []
        self.alpha = learning_rate
        self.fitness_func = error_func

    def update_weights(self):
        """
        1. This is where FSS should be used
        2. Each fish holds a weight vector
        3. We want the fish with the best fitness
        4.
        """
    def fit(self):
        pass

    def predict(self):
        pass