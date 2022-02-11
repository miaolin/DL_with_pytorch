import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# use torchtext version 0.4.0

import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

#train_data, valid_data, test_data = Multi30k(language_pair=('de', 'en'))
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N), N is the batch size
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        """
        x shape: (N) where N for batch size, we want it to be (1, N), seq_length is 1 because
        we are sending in a single word and not a sentence
        :param x:
        :return:
        """

        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]
        for t in range(1, target_len):
            # use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

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
    num_layers = 2
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

    model = seq2seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # what is this one?
    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        # model.eval()
        #
        # translated_sentence = translate_sentence(
        #     model, sentence, german, english, device, max_length=50
        # )
        #
        # print(f"Translated example sentence: \n {translated_sentence}")

        model.train()
        n = 0
        for batch in train_iterator:
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