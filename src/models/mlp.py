# mlp.py
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.activation_function = nn.ReLU()
        self.init_weights()

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation_function(x)
        x = self.output(x)
        return x

    def init_weights(self):
        nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity='relu') # TO NOTE
        nn.init.xavier_uniform_(self.output.weight) # TO NOTE

    def set_weights(self, weights_and_biases):
        hidden_weights_count = self.hidden.in_features * self.hidden.out_features

        hidden_weights = weights_and_biases[:hidden_weights_count]
        hidden_bias = weights_and_biases[hidden_weights_count:hidden_weights_count + self.hidden.out_features]

        output_weights_count = self.output.in_features * self.output.out_features
        output_weights = weights_and_biases[hidden_weights_count + self.hidden.out_features:
                                            hidden_weights_count + self.hidden.out_features + output_weights_count]
        output_bias = weights_and_biases[-self.output.out_features:]

        self.hidden.weight.data = torch.tensor(hidden_weights).view(self.hidden.out_features, self.hidden.in_features)
        self.hidden.bias.data = torch.tensor(hidden_bias)
        self.output.weight.data = torch.tensor(output_weights).view(self.output.out_features, self.output.in_features)
        self.output.bias.data = torch.tensor(output_bias)

    def set_hidden_layer(self, hidden_size, weights_and_biases=None):
        self.hidden = nn.Linear(self.input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        if weights_and_biases is not None:
            self.set_weights(weights_and_biases)


if __name__ == "__main__":
    # Example initialization
    input_layer_size = 10
    hidden_layer_size = 16

    model = MLP(input_layer_size, hidden_layer_size)
    print(model)

    print("Old weights")
    print(model.hidden.weight.data)
    print(model.hidden.bias.data)
    print(model.output.weight.data)
    print(model.output.bias.data)

    example_weights_biases = torch.randn((input_layer_size + 1) * hidden_layer_size + (hidden_layer_size + 1)).tolist()
    model.set_weights(example_weights_biases)

    print("New weights")
    print(model.hidden.weight.data)
    print(model.hidden.bias.data)
    print(model.output.weight.data)
    print(model.output.bias.data)
