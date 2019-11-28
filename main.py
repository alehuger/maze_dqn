from train_and_test import test_config
import numpy as np
from random_environment import Environment


if __name__ == "__main__":

    random_seed = 100
    np.random.seed(random_seed)
    seed_path = r'results/map_seed_' + str(random_seed)

    # Create a random environment
    current_environment = Environment(magnification=500)

    configuration_path = seed_path + '/test_'
    test_config(current_environment, configuration_path)
