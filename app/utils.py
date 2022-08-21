import numpy as np
import torch
from torch import nn
from torch.functional import F
import random

chars = 'abcdefghijklmnopqrstuvwxyz'

char_to_ix = {ch: i + 1 for i, ch in enumerate(chars)}
char_to_ix['<EOS>'] = 0

ix_to_char = {i: ch for ch, i in char_to_ix.items()}
ix_list = [i + 1 for i, ch in enumerate(chars)]


class Model(nn.Module):
    def __init__(self, _map, hidden_size, emb_dim=8, n_layers=1, dropout_p=0.2):
        """
        Input:
            _map: char_to_ix.
            hidden_size: Number of features to learn.
            emb_dim: Size of embedding vector.
            n_layers: Number of layers.
            dropout_p: Dropout probability.
        """
        super(Model, self).__init__()

        self.vocab_size = len(_map)
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.emb_dim)

        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True)

        self.dropout = nn.Dropout(self.dropout_p)

        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size)

    def forward(self, x, prev_state):
        """
        Input:
            x: x
            prev_state: The previous state of the model.

        Output:
            out: The output of the model.
            state: The current state of the model.
        """
        n_b, n_s = x.shape

        embed = self.embedding(x)
        yhat, state = self.lstm(embed, prev_state)

        yhat = self.dropout(yhat)
        out = self.fc(yhat)
        return out, state

    def init_state(self, b_size=1):
        return (torch.zeros(self.n_layers, b_size, self.hidden_size),
                torch.zeros(self.n_layers, b_size, self.hidden_size))


def load_model(path):
    m_data = torch.load(path)

    m = Model(
        _map=m_data["_map"],
        hidden_size=m_data["hidden_size"],
        emb_dim=m_data["emb_dim"],
        n_layers=m_data["n_layers"],
        dropout_p=m_data["dropout_p"])

    m.load_state_dict(m_data["state_dict"])
    l_hist = m_data["loss_history"]
    return m, l_hist


def keys_to_values(keys, _map, default):
    return [_map.get(key, default) for key in keys]


def sample_next(model, x, prev_state, topk=5, uniform=True):
    """
    Input:
        model: A Pytorch's nn.Module class.
        x: The input to the model.
        prev_state: The previous state of the model.
        topk: The top-k output to sample from. If None, sample from the entire output.
        uniform: Whether to sample from a uniform or a weighted distrubution of topk.

    Output:
        sampled_ix: The sampled index.
        state: The current state of the model.
    """
    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]

    # Get the top-k indexes and their values
    topk = topk if topk else last_out.shape[0]
    top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)

    # Get the softmax of the topk's and sample
    p = None if uniform else F.softmax(top_logit.detach(), dim=-1).numpy()
    sampled_ix = np.random.choice(top_ix, p=p)
    return sampled_ix, state


def sample(model, seed, topk=5, uniform=True, max_seqlen=18, stop_on=None, batched=True):
    """
    Input:
        model: A Pytorch's nn.Module class.
        seed: List of indexes to intialise model with.
        topk: The top-k output to sample from. If None, sample from the entire output.
        uniform: Whether to sample from a uniform or a weighted distrubution of topk.
        max_seqlen: The maximum sequence length to sample. 'seed' length is included.
        stop_on: Index that signals the end of sequence (sampling).
            If None, max_seqlen determines the end of sampling.

    Output:
        sampled_ix_list: List of sampled indexes.
    """
    seed = seed if isinstance(seed, (list, tuple)) else [seed]

    model.eval()
    with torch.no_grad():
        sampled_ix_list = seed[:]
        x = (torch.tensor([seed]), torch.tensor([len(seed)]))
        if not batched:
            x = torch.tensor([seed])

        prev_state = model.init_state(b_size=1)
        for t in range(max_seqlen - len(seed)):
            sampled_ix, prev_state = sample_next(model, x, prev_state, topk, uniform)

            sampled_ix_list.append(sampled_ix)
            x = (torch.tensor([[sampled_ix]]), torch.tensor([1]))
            if not batched:
                x = torch.tensor([[sampled_ix]])

            if sampled_ix == stop_on:
                break

    model.train()
    return sampled_ix_list


def try_parse_int(inp, default):
    if not inp:
        return default
    try:
        return int(inp)
    except:
        return default


MODEL_PATH = 'app/dinos.pt'
model, loss_history = load_model(MODEL_PATH)


def get_prediction(starts_with, amount):
    amount = try_parse_int(amount, 5)
    predictions = []

    for i in range(amount):
        word = list(starts_with.strip().lower()) if starts_with else []
        seed = keys_to_values(word, char_to_ix, char_to_ix['<EOS>']) if len(word) else random.choice(ix_list)

        prediction = keys_to_values(
            sample(model, seed, 5, False, 30, char_to_ix['<EOS>'], False),
            ix_to_char,
            '<?>'
        )

        readable_prediction = "".join(prediction[:-1]).capitalize()
        predictions.append(readable_prediction)

    return {'data': predictions}
