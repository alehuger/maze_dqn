import torch


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(
            in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=128)
        self.layer_3 = torch.nn.Linear(in_features=128, out_features=64)

        self.output_layer = torch.nn.Linear(
            in_features=64, out_features=output_dimension)

    def forward(self, net_input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(net_input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))

        output = self.output_layer(layer_3_output)
        return output
