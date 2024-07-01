# sparse auto-encoder in torch

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import islice

# 1. project embeddings to full feature dimension
# 2. sparsify by selecting features above a certain threshold (or just top k?)
# 3. reconstruct embeddings with matmul or weighted lookup table

# maps from embedding to dense weights
class Encoder(nn.Module):
    def __init__(self, embed_size, num_features):
        super().__init__()
        self.linear = nn.Linear(embed_size, num_features)

    def forward(self, embed):
        weights = self.linear(embed)
        return F.sigmoid(weights)

# maps from weights to reconstructed embedding
class Decoder(nn.Module):
    def __init__(self, embed_size, num_features):
        super().__init__()
        self.lookup = nn.Embedding(num_features, embed_size)

    # feats: [..., sel_feats]
    # weights: [..., sel_feats]
    def forward(self, feats, weights):
        vecs = self.lookup(feats) # [..., sel_feats, embed_size]
        return (weights.unsqueeze(-1)*vecs).sum(-2)

class AutoEncoder(nn.Module):
    def __init__(self, embed_size, num_features, topk):
        super().__init__()
        self.topk = topk
        self.encoder = Encoder(embed_size, num_features)
        self.decoder = Decoder(embed_size, num_features)

    def forward(self, embed):
        # do dense encoding
        project = self.encoder(embed)

        # sparsifry and index
        topk = project.topk(self.topk, dim=-1)
        feats, weights = topk.indices, topk.values

        # run decoder
        embed1 = self.decoder(feats, weights)

        return embed1

def dataset_csv(path, column, batch_size=32):
    datf = pd.read_csv(path, usecols=[column])
    text = datf.dropna()[column].tolist()
    data = torch.utils.data.DataLoader(
        text, batch_size=batch_size, shuffle=True
    )
    return data

def train(model, aenc, data, epochs=10, max_steps=None):
    # set up optimizer
    optim = torch.optim.Adam(aenc.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # set up loss function
    def run_loss(batch):
        with torch.no_grad():
            embed = model(batch)
        embed1 = aenc(embed)
        return loss_fn(embed, embed1)

    # evaluate some batches
    def run_eval(ne):
        with torch.no_grad():
            return sum([
                -run_loss(batch).item() for batch in islice(data, ne)
            ]) / ne

    # run some data epochs
    for e in range(epochs):
        # standard gradient update
        for step, batch in enumerate(tqdm(data, desc=f'EPOCH {e}')):
            optim.zero_grad()
            loss = run_loss(batch)
            loss.backward()
            optim.step()

            # break if we hit max steps
            if max_steps is not None and step >= max_steps:
                break

        # print out loss
        eval_loss = run_eval(16)
        print(f'LOSS: {eval_loss}')

