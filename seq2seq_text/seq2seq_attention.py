import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from utils import translate_sentence

from data_utils import english, german, train_data, valid_data, test_data

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (seq_length, N), N is the batch size
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_stats, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size*2)
        # hidden shape: (2, N, hidden_size)

        # concatenate hidden/cell states in the last dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return encoder_stats, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout=p)

        # hidden states from encoder and from previous time step in the decoder
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        """
        x shape: (N) where N for batch size, we want it to be (1, N), seq_length is 1 because
        we are sending in a single word and not a sentence
        :param x:
        :return:
        """
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        # h_reshaped: (17,1,1024), encoder_states: (17,1,2048)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention = self.softmax(energy)
        # (seq_length, N, 1) -> (N, 1, seq_length)
        attention = attention.permute(1,2,0)
        # N, 1, seq_length

        # encoder_states: (seq_length, N, hidden_size*2) -> (N, seq_length, hidden_size*2)
        encoder_stats = encoder_states.permute(1,0,2)

        # bmm: (N, 1, seq_length) * (N, seq_length, hidden_size*2) -> (N, 1, hidden_size*2)
        # (N, 1, hidden_size * 2) --> (1, N, hidden_size*2)
        context_vector = torch.bmm(attention, encoder_stats).permute(1,0,2)

        # (N, 1, hidden_size*2) + (N, 1, embedding) -> (1, N, hidden_size*2+embedding)
        rnn_input = torch.cat((context_vector, embedding), dim=2)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class seq2seqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seqAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # encoder_stats: (seq_length, batch, hidden_size*2)
        # hidden: (1, batch, hidden_size)
        # cell: (1, batch, hidden_size)
        encoder_stats, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]
        for t in range(1, target_len):
            # use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, encoder_stats, hidden, cell)

            # store next output prediction
            outputs[t] = output

            # get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # get the next input based on teacher_force_ratio
            prob = random.random()
            if prob < teacher_force_ratio:
                x = target[t]
            else:
                x = best_guess
        return outputs


if __name__ == "__main__":
    # Training hyperparameters
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 64

    #Model hyperparameters
    load_model = False
    input_size_encoder = len(german.vocab)
    input_size_decoder = len(english.vocab)
    output_size = len(english.vocab)
    encoder_embedding_size = 300
    decoder_embeeding_size = 300
    hidden_size = 1024
    num_layers = 1
    enc_dropout = 0.5
    dec_dropout = 0.5

    writer = SummaryWriter(f"runs/loss_plot")
    step = 0

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device,
    )

    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    print(encoder_net)

    decoder_net = Decoder(input_size_decoder, decoder_embeeding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    print(decoder_net)

    model = seq2seqAttention(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # what is this one?
    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        model.eval()

        translated_sentence = translate_sentence(
            model, sentence, german, english, device, max_length=50
        )
        print(f"Translated example sentence: \n {translated_sentence}")

        n = 0
        for batch in train_iterator:
            model.train()
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(inp_data, target)

            # output shape: (trg_len, batch_size, output_dim)
            # remove start token
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # backprop
            loss.backward()

            # clip to avoid exploding gradient issues
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # gradient descent step
            optimizer.step()

            #
            writer.add_scalar("training loss", loss, global_step=step)
            step += 1

            print([step, loss.item()])

            if step % 10 == 0:
                model.eval()
                translated_sentence = translate_sentence(
                    model, sentence, german, english, device, max_length=50
                )
                print(f"Translated example sentence: \n {translated_sentence}")
