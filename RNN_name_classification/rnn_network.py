import torch
import torch.nn as nn

from letter_utils import *


class NameClassifyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameClassifyRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.i2o = nn.Linear(in_features=input_size + hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


if __name__ == "__main__":
    n_hidden = 128
    rnn = NameClassifyRNN(n_letters, n_hidden, 18)

    input = letter_to_tensor('A')
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input, hidden)

    input = line_to_tensor('Albert')
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input[0], hidden)
    print(output)