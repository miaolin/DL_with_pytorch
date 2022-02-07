import random
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim

from seq2seq_attention.data import MAX_LENGTH, SOS_token, EOS_token, tensorsFromPair, prepare_data
from seq2seq_attention.seq2seq_model import EncoderRNN, DecoderRNN, AttnDecoderRNN
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          device, max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # teacher forcing: feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] # teacher forcing

    else:
        # without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach() # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(input_lang, output_lang, encoder, decoder, n_iters, device, print_every=1000, plot_every=100, learning_rate=0.01):

    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_paris = [tensorsFromPair(input_lang, output_lang, random.choice(pairs), device=device) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        train_paris = training_paris[iter - 1]
        input_tensor = train_paris[0]
        target_tensor = train_paris[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                     device, max_length=MAX_LENGTH)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % ("aaa", iter, iter/n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

    hidden_size = 256
    encoder1 = EncoderRNN(input_size=input_lang.n_words, hidden_size=hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size=hidden_size, output_size=output_lang.n_words, dropout_p=0.1).to(device)

    train_iters(input_lang=input_lang, output_lang=output_lang, encoder=encoder1, decoder=attn_decoder1, n_iters=75000, device=device)