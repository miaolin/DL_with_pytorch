import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import string
import random
import sys
import unidecode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open("data/names.txt").read())


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class TextGenerator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 50
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.03

    def char_tensor(self, ch_string):
        ch_tensor = torch.zeros(len(ch_string)).long()
        for c in range(len(ch_string)):
            ch_tensor[c] = all_characters.index(ch_string[c])
        return ch_tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file)-self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx: end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])
        return text_input.long(), text_target.long()

    def generate(self, initial_str="A", predict_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)
        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)
        return predicted

    def train(self):
        self.rnn = TextRNN(input_size=n_characters, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           output_size=n_characters).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f"runs/names0") # for tensorboard

        print("start training")
        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            inp = inp.to(device)
            target = target.to(device)

            # why need to initial at each epoch
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            # what is the different between rnn.zero_grad() vs optimizer.zero_grad()
            self.rnn.zero_grad()
            loss = 0
            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f"Loss: {loss}")
                print(self.generate())

            writer.add_scalar("Training loss", loss, global_step=epoch)


if __name__ == "__main__":
    gennames = TextGenerator()
    gennames.train()



