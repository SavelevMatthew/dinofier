import torch
from torch import nn
torch.set_num_threads(1)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim=8, n_layers=1, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(
          input_size=self.embed_dim,
          hidden_size=self.hidden_size,
          num_layers=self.n_layers,
          batch_first=True
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, prev_state=None):
        embedded = self.embedding(x)
        y, state = self.lstm(embedded, prev_state)
        y = self.dropout(y)
        y = self.fc(y)
        return y, state
