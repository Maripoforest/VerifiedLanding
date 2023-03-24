import torch
import torch.nn as nn

def alpha_crown(network, x, epsilon, alpha):
    input_bounds = (torch.clamp(x - epsilon, min=0.0), torch.clamp(x + epsilon, max=1.0))
    lower_bounds, upper_bounds = input_bounds

    for layer in network:
        if isinstance(layer, nn.Linear):
            W, b = layer.weight, layer.bias
            new_lower_bounds = linear_layer_propagation(W, b, lower_bounds, upper_bounds, is_lower=True)
            new_upper_bounds = linear_layer_propagation(W, b, lower_bounds, upper_bounds, is_lower=False)

        elif isinstance(layer, nn.ReLU):
            new_lower_bounds, new_upper_bounds = relu_linear_relaxation(lower_bounds, upper_bounds, alpha)

        lower_bounds, upper_bounds = new_lower_bounds, new_upper_bounds

    return lower_bounds, upper_bounds

def linear_layer_propagation(W, b, lower_bounds, upper_bounds, is_lower):
    if is_lower:
        new_bounds = W.clamp(min=0) @ lower_bounds + W.clamp(max=0) @ upper_bounds + b.unsqueeze(1)
    else:
        new_bounds = W.clamp(min=0) @ upper_bounds + W.clamp(max=0) @ lower_bounds + b.unsqueeze(1)

    return new_bounds

def relu_linear_relaxation(lower_bounds, upper_bounds, alpha):
    slope = (upper_bounds - lower_bounds) / (alpha * (upper_bounds - lower_bounds) + (1 - alpha) * (lower_bounds + upper_bounds))
    bias = lower_bounds - slope * lower_bounds

    new_lower_bounds = slope * lower_bounds + bias
    new_upper_bounds = slope * upper_bounds + bias

    return new_lower_bounds, new_upper_bounds

# Example usage:
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNN()
x = torch.tensor([[0.5, 0.7]])
epsilon = 0.1
alpha = 0.5
lower_bounds, upper_bounds = alpha_crown(model.layers, x.t(), epsilon, alpha)

print("Lower bounds:", lower_bounds)
print("Upper bounds:", upper_bounds)
