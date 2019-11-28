from network import Network
import torch
import numpy as np


class DQN:

    def __init__(self, output_dim, learning_rate=0.001, gamma=0.9):
        self.q_target_network = Network(
            input_dimension=2, output_dimension=output_dim)
        self.q_network = Network(
            input_dimension=2,
            output_dimension=output_dim)
        self.gamma = gamma
        self.optimiser = torch.optim.Adam(
            self.q_network.parameters(), lr=learning_rate)

        self.magnification = 1000
        self.image = np.zeros([self.magnification, self.magnification, 3],
                              dtype=np.uint8)
        self.decaying_rate = 0.003 / 200

    def update_target_network(self):
        self.q_target_network.load_state_dict(self.q_network.state_dict())

    def update_learning_rate(self):
        for param_group in self.optimiser.param_groups:
            learning_rate = param_group['lr']
            learning_rate = max(0.001, learning_rate - self.decaying_rate)
            param_group['lr'] = learning_rate

    def train_q_network(self, batch):
        self.optimiser.zero_grad()
        delta_prediction, loss_avg = self._calculate_loss(batch)
        loss_avg.backward()
        self.optimiser.step()
        return delta_prediction

    def predict(self, example_array):
        input_tensor = torch.tensor(example_array).float()
        q_value_predicted = self.q_network.forward(
            input_tensor).detach().numpy()
        return np.argmax(q_value_predicted)

    def predict_target(self, example_array):
        input_tensor = torch.tensor(example_array).float()
        q_value_predicted = self.q_target_network.forward(
            input_tensor).detach().numpy()
        return np.argmax(q_value_predicted)

    def _calculate_loss(self, batch):

        states_tensor = torch.Tensor(np.array([t[0] for t in batch]))
        actions_tensor = torch.LongTensor(np.array([[t[1]] for t in batch]))
        rewards_tensor = torch.Tensor(np.array([[t[2]] for t in batch]))
        next_states_tensor = torch.Tensor(np.array([t[3] for t in batch]))

        q_values_predicted = self.q_network.forward(states_tensor)
        q_values_predicted_given_action = q_values_predicted.gather(
            dim=1, index=actions_tensor)

        q_values_predicted_next_state = self.q_target_network.forward(
            next_states_tensor).detach()

        q_values_next_state_max = torch.max(
            q_values_predicted_next_state, dim=1).values
        product = self.gamma * q_values_next_state_max.unsqueeze(1)

        rewards_gt = rewards_tensor + product

        size = batch.shape[0]
        eps = 0.01 * np.ones(size)
        delta = np.reshape(
            (q_values_predicted_given_action.detach().numpy() -
             rewards_gt.detach().numpy()),
            size)
        delta_prediction = np.abs(delta) + eps

        return delta_prediction, torch.nn.MSELoss()(
            q_values_predicted_given_action, rewards_gt)

    def visualize_dqn(self):
        grid_colors = np.zeros((100, 100), dtype='uint8')
        starting_state = np.array([0.01, 0.01])
        for x_inc in range(100):
            for y_inc in range(100):
                current_state = np.array(
                    starting_state + .01 * np.array([x_inc, y_inc]), dtype=np.double)
                grid_colors[y_inc, x_inc] = self.predict(current_state)

        return grid_colors

    def visualize_dqn_target(self):
        grid_colors = np.zeros((100, 100), dtype='uint8')
        starting_state = np.array([0.01, 0.01])
        for x_inc in range(100):
            for y_inc in range(100):
                current_state = np.array(
                    starting_state + .01 * np.array([x_inc, y_inc]), dtype=np.double)
                grid_colors[y_inc, x_inc] = self.predict_target(current_state)

        return grid_colors
