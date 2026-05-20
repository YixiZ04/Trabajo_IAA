"""
    Coded here the object used to build an SOM using minisom
"""


from minisom import MiniSom

RANDOM_SEED = 42

class HexagonalSOMBuilder:
    """
        Builds and trains a hexagonal Minisom object from minisom.
        Inputs:
        n_neurons: number of neurons. Default is 30.
        sigma: This is the neighborh to update. Default is 5
        learning_rate: default is 0.5
        activation_distance, this is the distance function. Default is "euclidean".
        topology: Can be "rectangular" or "hexagonal". But "hexagonal" is default.
        num_epocs: number of epochs. Default is 20000.
        random_seed: random seed. Default is 42
    """
    def __init__ (self,
                  n_neurons=30,
                  sigma=5,
                  learning_rate=0.5,
                  activation_distance="euclidean",
                  topology="hexagonal",
                  neighbordhood_function= "gaussian",
                  num_epochs = 20000,
                  random_seed=RANDOM_SEED):

        self.n_neurons = n_neurons
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.activation_distance = activation_distance
        self.topology = topology
        self.neighbordhood_function = neighbordhood_function
        self.num_epochs = num_epochs
        self.random_seed = random_seed

    def train_som(self, train_data) -> MiniSom:
        """
            Fit the training data to SOM. This should
        """
        image_shape = len(train_data[0])
        som = MiniSom(self.n_neurons, self.n_neurons,
                      image_shape,
                      sigma= self.sigma,
                      learning_rate= self.learning_rate,
                      activation_distance=self.activation_distance,
                      topology = self.topology,
                      neighborhood_function= self.neighbordhood_function,
                      random_seed=self.random_seed)
        som.train(train_data, self.num_epochs)
        return som

if __name__ == "__main__":
    my_som = HexagonalSOMBuilder()



