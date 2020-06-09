import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class EmbAttModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, out_dim):
        super(EmbAttModel, self).__init__()
        """ layers """
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.weight_layer = nn.Linear(emb_dim, 1)
        self.linear = nn.Linear(emb_dim, out_dim)


    def forward(self, inp, block_ids=None):                      # shape(inp)       : B x W
        emb_output = self.embedding_layer(inp)   # shape(emb_output): B x W x emd_dim
        weights = self.weight_layer(emb_output)  # shape(weights)   : B x W x 1
        weights = weights.squeeze(-1)            # shape(weights)   : B x W
        attentions = nn.Softmax(dim=-1)(weights) # shape(attention) : B x W
        #NOTE: ensure block_ids are right tensors
        if block_ids is not None:
            attentions = (1 - block_ids) * attentions

        context = torch.einsum('bw,bwe->be',
                        [attentions, emb_output])# shape(context)   : B x W
        out = self.linear(context)               # shape(out)       : B X out_dim
        return out, attentions

    def get_embeddings(self, inp):
        emb_output = self.embedding_layer(inp)
        return emb_output

    def get_linear_wts(self):
        return self.linear.weight, self.linear.bias

    def get_final_states(self, inp):
        emb_output = self.embedding_layer(inp)
        return emb_output


class BiLSTMAttModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super(BiLSTMAttModel, self).__init__()
        """ layers """
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.lstm_layer = nn.LSTM(emb_dim, hid_dim, bidirectional=True,
                            batch_first=True)
        self.weight_layer = nn.Linear(2*hid_dim, 1)
        self.linear = nn.Linear(2*hid_dim, out_dim)


    def forward(self, inp, block_ids=None):                      # shape(inp)       : B x W
        emb_output = self.embedding_layer(inp)   # shape(emb_output): B x W x emd_dim
        lstm_hs, _ = self.lstm_layer(emb_output) # shape(lstm_hs)   : B x W x 2*hid_dim
        weights = self.weight_layer(lstm_hs)     # shape(weights)   : B x W x 1
        weights = weights.squeeze(-1)            # shape(weights)   : B x W
        attentions = nn.Softmax(dim=-1)(weights) # shape(attention) : B x W

        #NOTE: ensure block_ids are right tensors
        if block_ids is not None:
            attentions = (1 - block_ids) * attentions

        context = torch.einsum('bw,bwe->be',
                        [attentions, lstm_hs])# shape(context)   : B x W
        out = self.linear(context)               # shape(out)       : B X out_dim
        return out, attentions

    def get_embeddings(self, inp):
        emb_output = self.embedding_layer(inp)
        return emb_output

    def get_linear_wts(self):
        return self.linear.weight, self.linear.bias

    def get_final_states(self, inp):
        emb_output = self.embedding_layer(inp)
        lstm_hs, _ = self.lstm_layer(emb_output)
        return lstm_hs


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim):
        super(BiLSTMModel, self).__init__()
        """ layers """
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim)
        self.lstm_layer = nn.LSTM(emb_dim, hid_dim, bidirectional=True,
                            batch_first=True)
        self.linear = nn.Linear(2*hid_dim, out_dim)


    def forward(self, inp):                       # shape(inp)       : B x W
        emb_output = self.embedding_layer(inp)    # shape(emb_output): B x W x emd_dim
        lstm_hns, _ = self.lstm_layer(emb_output) # shape(lstm_hs)   : B x W x 2*hid_dim
        B, W, h = lstm_hns.size()
        output = lstm_hns.view(B, W, 2, h//2)

        #NOTE: last states is concat of fwd lstm, and bwd lstm
        #      0 refers to fwd direction, and 1 for bwd direction
        #      https://pytorch.org/docs/stable/nn.html?highlight=lstm#lstm
        last_states = torch.cat((output[:, -1, 0, :], output[:, 0, 1, :]), -1)

        out = self.linear(last_states)

        #NOTE: uniform attention is returned for consistency
        #      w/ other modules which return attention weights
        attns = 1.0/ W * torch.ones((B, W))

        return out, attns