import torch
import random
import numpy as np
from flask import jsonify
from model import LSTMModel
from torch.functional import F

model = LSTMModel(27, 128, 16, 2, 0.2)
model.load_state_dict(torch.load('model.pt'))
model.eval()

EOS = '<EOS>'
chars = [EOS] + [chr(97 + i) for i in range(26)]
ch_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def get_starter(prefix):
    if not prefix:
        return [random.choice(range(len(chars) - 1)) + 1]
    else:
        return [ch_to_idx.get(ch, 1) for ch in prefix.lower()]


def generate_next(x, prev_state, top_k, uniform):
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]
    top_logit, top_idx = torch.topk(last_out, top_k)

    probs = None if uniform else F.softmax(top_logit.detach(), dim=-1).numpy()

    next_idx = np.random.choice(top_idx, p=probs)

    return next_idx, state


def generate(start, max_len, top_k, uniform):
    idx = start[:]
    with torch.no_grad():
        x = torch.tensor([start])
        state = None

        for i in range(max_len - len(start)):
            next_idx, state = generate_next(x, state, top_k, uniform)
            idx.append(next_idx)
            x = torch.tensor([[next_idx]])

            if next_idx == ch_to_idx[EOS]:
                idx.pop()
                break

    return idx


def get_predictions(prefix, amount, max_len, uniform, top_k):
    results = []
    for i in range(amount):
        starter = get_starter(prefix)
        idx = generate(starter, max_len, top_k, uniform)
        name = ''.join([idx_to_char[j] for j in idx])
        results.append(name)
    return jsonify(results)
