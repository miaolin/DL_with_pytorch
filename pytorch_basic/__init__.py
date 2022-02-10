import torch
import torchvision
import torch.nn.functional as F # paramterless functions
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm # for a nice progress bar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# use it as many-to-one
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.zeros(0), self.hidden_size).to(device)

        # forward progagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # decode the hidden state of the last time step
        out = self.fc(out)
        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # decode the hidden state of the last time step
        out = self.fc(out)
        return out


if __name__ == "__main__":
    input_size = 2
    hidden_size = 256
    num_layers = 2
    num_classes = 10
    sequence_length = 28
    learning_rate = 0.005
    batch_size = 64
    num_epochs = 3

    # load data
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_load = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)

            output = model(data)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()






