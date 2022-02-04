import torch
import torch.nn as nn

from math import sqrt
import numpy as np
from numpy.random import rand
from numpy.random import uniform

torch.manual_seed(1)


def xavier_initial(input_size, param_num):
    lower, upper = - 1.0 / sqrt(input_size), 1.0 / sqrt(input_size)
    numbers = rand(param_num)
    scalec = lower + numbers * (upper - lower)

    scales = uniform(low=lower, high=upper, size=param_num)
    return scalec


if __name__ == "__main__":
    scalec = xavier_initial(input_size=10, param_num=1000)

    # rnn cell test
    rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=False)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
    print("sequence first rnn")
    print(output.shape)
    print(hn.shape)

    # rnn cell test
    rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    input = torch.randn(3, 5, 10) # batch * seq * input_features
    h0 = torch.randn(2, 3, 20) #
    output, hn = rnn(input, h0)
    print("batch first rnn")
    print(output.shape)
    print(hn.shape)

    # lstm cell test
    rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=False)
    input = torch.randn(5, 3, 10) # seq*batch*input_features
    h0 = torch.randn(2, 3, 20) # (dire*layer)*batch*hidden
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print("seq first lstm")
    print(output.shape)
    print(hn.shape)
    print(cn.shape)

    # lstm cell test
    rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    input = torch.randn(3, 5, 10) # batch*seq*input_features
    h0 = torch.randn(2, 3, 20) # (dire*layer)*batch*hidden
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print("batch first lstm")
    print(output.shape)
    print(hn.shape)
    print(cn.shape)
