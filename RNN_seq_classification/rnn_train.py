import torch
import torch.nn as nn

from res.sequential_tasks import TemporalOrderExp6aSequence as QRSU
from res.plot_lib import set_default, plot_state, print_colourbar


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity="relu", batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.rnn(x)[0]
        x = self.linear(h)
        return x


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.lstm(x)[0]
        x = self.linear(h)
        return x

    def get_states_across_time(self, x):
        h_c = None
        h_list, c_list = list(), list()
        with torch.no_grad():
            for t in range(x.size(1)): # sequence length
                h_c = self.lstm(x[:, [t], :], h_c)[1]
                h_list.append(h_c[0])
                c_list.append(h_c[1])
            h = torch.cat(h_list)
            c = torch.cat(c_list)
        return h, c


def train(model, train_data_gen, criterion, optimizer, device):

    model.train()
    num_correct = 0
    for batch_idx in range(len(train_data_gen)):
        # get batch data and send to device
        data, target = train_data_gen[batch_idx]
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

        if batch_idx == 0:
            print("train data shape is ")
            print(data.shape)

        # perform forward pass
        output = model(data)

        # pick the output corresponding to last sequence element
        output = output[:, -1, :]
        target = target.argmax(dim=1)

        loss = criterion(output, target)

        # clear the gradient buffers of the optimized parameters.
        # otherwise gradients from the previous batch would be accumulated
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        y_pred = output.argmax(dim=1)
        num_correct += (y_pred == target).sum().item()

    return num_correct, loss.item()


def test(model, test_data_gen, criterion, device):

    model.eval()

    num_correct = 0

    print(len(test_data_gen))
    # A context manager is used to disable gradient calculations during inference
    # to reduce memory usage, as we typically don't need the gradients at this point
    with torch.no_grad():
        for batch_idx in range(len(test_data_gen)):
            data, target = test_data_gen[batch_idx]
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).long().to(device)

            if batch_idx == 0:
                print("test data shape is ")
                print(data.shape)

            output = model(data)
            # only get the output corresonding to the last sequence element
            output = output[:, -1, :]

            target = target.argmax(dim=1)
            loss = criterion(output, target)

            y_pred = output.argmax(dim=1)
            num_correct += (y_pred == target).sum().item()

    return num_correct, loss.item()


def train_and_test(model, train_data_gen, test_data_gen, criterion, optimizer, max_epochs, verbose=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    historical_train = {'loss': [], 'acc': []}
    historical_test = {'loss': [], 'acc': []}

    for epoch in range(max_epochs):
        num_correct, loss = train(model, train_data_gen, criterion, optimizer, device)
        accuracy = float(num_correct) / (len(train_data_gen) * train_data_gen.batch_size) * 100
        historical_train['loss'].append(loss)
        historical_train['acc'].append(accuracy)

        # do the same for test loop
        num_correct, loss = test(model, test_data_gen, criterion, device)
        accuracy = float(num_correct) / (len(test_data_gen) * test_data_gen.batch_size) * 100
        historical_test['loss'].append(loss)
        historical_test['acc'].append(accuracy)

        if verbose or epoch + 1 == max_epochs:
            print(f'[Epoch {epoch + 1}/{max_epochs}'
                  f" loss: {historical_train['loss'][-1]:.4f}, acc: {historical_train['acc'][-1]:2.2f}%"
                  f" - test_loss: {historical_test['loss'][-1]:.4f}, test_acc: {historical_test['acc'][-1]:2.2f}%")

    return model


if __name__ == '__main__':
    difficulty = QRSU.DifficultyLevel.EASY
    batch_size = 32
    train_data_gen = QRSU.get_predefined_generator(difficulty, batch_size)
    test_data_gen = QRSU.get_predefined_generator(difficulty, batch_size)

    # setup RNN and training settings
    input_size = train_data_gen.n_symbols
    hidden_size = 4
    output_size = train_data_gen.n_classes

    # model = SimpleRNN(input_size, hidden_size, output_size)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    # max_epochs = 100
    # model = train_and_test(model, train_data_gen, test_data_gen, criterion, optimizer, max_epochs)

    model = SimpleLSTM(input_size, hidden_size, output_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    max_epochs = 10
    model = train_and_test(model, train_data_gen, test_data_gen, criterion, optimizer, max_epochs)

    # visualize LSTM
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        data = test_data_gen[0][0]
        X = torch.from_numpy(data).float().to(device)
        H_t, C_t = model.get_states_across_time(X)

    print("Color range is as follows:")
    print_colourbar()

    plot_state(X, C_t, b=9, decoder=test_data_gen.decode_x)
    plot_state(X, H_t, b=9, decoder=test_data_gen.decode_x)
