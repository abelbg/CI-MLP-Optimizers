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
        nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity='relu')  # He initialization
        nn.init.xavier_uniform_(self.output.weight)

    def set_weights(self, weights_and_biases):
        # Debugging: Print the total size of weights and biases provided
        # print("Total size of weights_and_biases:", len(weights_and_biases))

        # Calculate the split indices for weights and biases for the hidden layer
        hidden_weights_count = self.hidden.in_features * self.hidden.out_features
        # print("Expected hidden weights count:", hidden_weights_count)

        # Split the list into parts for each weight and bias
        hidden_weights = weights_and_biases[:hidden_weights_count]
        hidden_bias = weights_and_biases[hidden_weights_count:hidden_weights_count + self.hidden.out_features]

        # Debugging: Print sizes of hidden weights and biases
        # print("Actual hidden weights size:", len(hidden_weights))
        # print("Actual hidden biases size:", len(hidden_bias))

        # Calculate indices for the output layer
        output_weights_count = self.output.in_features * self.output.out_features
        # print("Expected output weights count:", output_weights_count)

        output_weights = weights_and_biases[hidden_weights_count + self.hidden.out_features:
                                            hidden_weights_count + self.hidden.out_features + output_weights_count]
        output_bias = weights_and_biases[-self.output.out_features:]

        # Debugging: Print sizes of output weights and biases
        # print("Actual output weights size:", len(output_weights))
        # print("Actual output biases size:", len(output_bias))

        # Reshape and set the weights and biases
        self.hidden.weight.data = torch.tensor(hidden_weights).view(self.hidden.out_features, self.hidden.in_features)
        self.hidden.bias.data = torch.tensor(hidden_bias)
        self.output.weight.data = torch.tensor(output_weights).view(self.output.out_features, self.output.in_features)
        self.output.bias.data = torch.tensor(output_bias)

    def set_hidden_layer(self, hidden_size):
        self.hidden = nn.Linear(self.input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)


if __name__ == "__main__":
    # Example initialization
    input_layer_size = 10   # Number of input features
    hidden_layer_size = 16  # Number of neurons in hidden layer

    model = MLP(input_layer_size, hidden_layer_size)
    print(model)

    print("Old")
    print(model.hidden.weight.data)
    print(model.hidden.bias.data)
    print(model.output.weight.data)
    print(model.output.bias.data)

    # Example weights and biases list (random values for demonstration)
    example_weights_biases = torch.randn((input_layer_size + 1) * hidden_layer_size +
                                         (hidden_layer_size + 1)).tolist()
    model.set_weights(example_weights_biases)

    print("New")
    print(model.hidden.weight.data)
    print(model.hidden.bias.data)
    print(model.output.weight.data)
    print(model.output.bias.data)
