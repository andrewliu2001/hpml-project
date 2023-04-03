import torch

# Define hyperparameters
input_size = 10
hidden_size = 20

# Define custom Attention module
class CustomAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomAttention, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.linear2 = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, hidden):
        energy = torch.tanh(self.linear1(inputs) + self.linear1(hidden))
        attention = torch.softmax(self.linear2(energy), dim=1)
        context = torch.sum(inputs * attention, dim=1)
        return context

# Define input tensor
inputs = torch.randn(1, 5, input_size)

# Create custom Attention instance
attention = CustomAttention(input_size, hidden_size)

# Compute attention
hidden = torch.randn(1, hidden_size)
context = attention(inputs, hidden)

# Print output
print("Context:", context)