import numpy as np

from replayBuffer import ReplayBuffer
from dqn import DQN


def get_valid_actions():
    valid_actions = [[0, 0.01], [0.01, 0], [0, -0.01], [-0.01, 0]]
    return np.array(valid_actions, dtype=np.float32)


class Agent:

    # Function to initialise the agent
    def __init__(self, init_state, config_path):

        # Set the episode length (you will need to increase this)
        self.episode_length = 140
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.total_reward = None
        self.actions2direction = get_valid_actions()
        self.nb_actions = self.actions2direction.shape[0]

        self.dqn = DQN(self.nb_actions, learning_rate=0.005, gamma=0.9)
        self.batch_size = 128
        self.replay_buffer = ReplayBuffer(self.batch_size)
        self.epsilon = 1
        self.delta = 0
        self.nb_episodes = 0
        self.episode_increase = 0.5
        self.hit_wall = False
        self.hit_wall_init = False
        self.epsilon_end = 0
        self.initialisation_steps = - self.episode_increase * 3
        self.weight_buffer = []
        self.transition_probabilities = []
        self.init_state = init_state
        self.config_path = config_path
        self.final_distance = None

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % int(self.episode_length) == 0 and bool(self.num_steps_taken):
            print("Exploration steps : ", self.num_steps_taken - self.initialisation_steps)
            self.epsilon_end = self.epsilon
            self.initialisation_steps += (1 - int(self.hit_wall_init)) * self.episode_increase

            # New episode
            self.episode_length += (1 - int(self.hit_wall_init)) * self.episode_increase
            self.initialisation_steps = min(180, self.initialisation_steps)
            self.num_steps_taken = 0
            self.hit_wall = False
            self.hit_wall_init = False
            self.final_distance = None
            self.nb_episodes += 1
            self.delta = 0.7 / (self.episode_length - max(0, self.initialisation_steps))
            self.epsilon = 1
            self.dqn.update_learning_rate()
            self.dqn.update_target_network()

            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        self.num_steps_taken += 1
        if self.num_steps_taken < self.initialisation_steps:
            if self.hit_wall:
                # print("Hit Wall during initialisation, init_steps: ", self.initialisation_steps)
                self.hit_wall_init = True
                action = self._epsilon_greedy_policy(0.25, state)
            else:
                action = self._epsilon_greedy_policy(0, state)

        elif self.num_steps_taken == self.initialisation_steps:
            print('Initialisation completed! steps : ', self.initialisation_steps)
            action = self._epsilon_greedy_policy(self.epsilon, state)

        else:
            action = self._epsilon_greedy_policy(self.epsilon, state)
            self.epsilon -= self.delta

        self.state = state
        self.action = action
        continuous_action = self._discrete_action_to_continuous(action)
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        init_flag = not bool(self.nb_episodes)

        self.final_distance = distance_to_goal
        if np.allclose(next_state, self.state):
            self.hit_wall = True

        reward = self._custom_reward(next_state, distance_to_goal)

        transition = (self.state, self.action, reward, next_state)
        self.replay_buffer.add(transition)

        if not init_flag or self.num_steps_taken > self.batch_size:
            mini_batch = self.replay_buffer.random_sampling(False)
            delta_prediction = self.dqn.train_q_network(mini_batch)
            self.replay_buffer.update_weight_buffer(delta_prediction)

        elif self.num_steps_taken > 32:
            mini_batch = self.replay_buffer.random_sampling(True)
            self.dqn.train_q_network(mini_batch)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # You should change it so that it returns the action with the highest Q-value
        action = self.dqn.predict(state)
        c_action = self._discrete_action_to_continuous(action)
        next_state = state + c_action
        next_state_action = self.dqn.predict(next_state)
        if action == next_state_action:
            return 2 * np.around(c_action, 2)
        return c_action

    def _get_greedy_discrete_action(self, state):
        action = self.dqn.predict(state)
        return action

    def _discrete_action_to_continuous(self, discrete_action):
        continuous_action = self.actions2direction[discrete_action]
        return continuous_action

    def _epsilon_greedy_policy(self, epsilon, state):
        if np.random.random() < epsilon:
            return self._random_step()
        else:
            return self._get_greedy_discrete_action(state)

    def _random_step(self):
        return np.random.randint(0, self.nb_actions)

    def _custom_reward(self, next_state, distance_to_goal):
        if np.allclose(self.state, next_state):
            return (np.sqrt(2) - distance_to_goal)/2
        else:
            return (np.sqrt(2) - distance_to_goal) / np.sqrt(2)
