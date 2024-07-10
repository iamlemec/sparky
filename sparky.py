# sparse auto-encoder in torch

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from math import sqrt
from tqdm import tqdm
from itertools import islice

# cuda viable bincount function (x is flattened)
def bincount(x, size):
    total = torch.zeros(size, dtype=torch.int64, device=x.device)
    ones = torch.ones(x.numel(), dtype=torch.int64, device=x.device)
    return total.scatter_add_(0, x.reshape(-1), ones)

# load in csv text dataset by column name
def dataset_csv(path, column, batch_size=32):
    datf = pd.read_csv(path, usecols=[column])
    text = datf.dropna()[column].tolist()
    return torch.utils.data.DataLoader(text, batch_size=batch_size, shuffle=True)

# maps from embedding to dense weights
# this is single layer for now but could be higher
class Encoder(nn.Module):
    def __init__(self, embed_size, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(embed_size))
        self.weight = nn.Parameter(torch.empty(num_features, embed_size))

    def forward(self, embed):
        embed0 = embed - self.bias
        weights = F.linear(embed0, self.weight)
        return weights

# maps from (sparse) weights to reconstructed embedding
class Decoder(nn.Module):
    def __init__(self, embed_size, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.empty(embed_size))
        self.lookup = nn.Embedding(num_features, embed_size)

    # feats: [batch_size, sel_feats]
    # weights: [batch_size, sel_feats]
    def forward(self, feats, weights, bias=True):
        vecs = self.lookup(feats) # [batch_size, sel_feats, embed_size]
        embed = torch.matmul(vecs.mT, weights.unsqueeze(-1)).squeeze(-1)
        if bias:
            embed += self.bias
        return embed

class AutoEncoder(nn.Module):
    def __init__(self, embed_size, num_features, topk, dead_cutoff=100_000, dead_topk=512, pin_bias=True):
        super().__init__()
        self.num_features = num_features
        self.topk = topk

        self.dead_cutoff = dead_cutoff
        self.dead_topk = dead_topk

        self.encoder = Encoder(embed_size, num_features)
        self.decoder = Decoder(embed_size, num_features)

        self.register_buffer('last_usage', torch.zeros(num_features, dtype=torch.int64))

        self.init_weights()
        if pin_bias:
            self.decoder.bias = self.encoder.bias

    def init_weights(self):
        self.encoder.bias.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.encoder.weight.data.normal_(0, 0.1)
        self.decoder.lookup.weight.data.normal_(0, 0.1)

    def reset_usage(self):
        self.last_usage.fill_(0)

    def features(self, embed):
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1)
        return feats

    # embed: [batch_size, embed_size]
    def forward(self, embed, undead=False):
        # sparsifry and index
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1)

        # accumulate usage stats
        with torch.no_grad():
            batch_size, _ = embed.shape
            total = bincount(feats, self.num_features)
            self.last_usage += batch_size # increment all
            self.last_usage *= 1 - total.clamp(max=1) # zero out used

        # run decoder
        recon = self.decoder(feats, weights)
        embed1 = F.normalize(recon)

        # add in resurrected feats
        if undead:
            # find dead_topk least used features
            dead_mask = self.last_usage > self.dead_cutoff
            dead_total = dead_mask.sum().item()
            dead_project = project[:, dead_mask]

            # reconstruct undead embeddings
            undead_topk = np.minimum(self.dead_topk, dead_total)
            undead_weights, undead_feats = dead_project.topk(undead_topk, dim=-1)
            undead_recon = self.decoder(undead_feats, undead_weights, bias=False)
            return embed1, undead_recon
        else:
            return embed1

class Trainer:
    def __init__(self, embed, model, undead_weight=0.03):
        self.embed = embed
        self.model = model
        self.undead_weight = undead_weight
        self.loss_fn = torch.nn.MSELoss()

    def features(self, text):
        with torch.no_grad():
            vecs = self.embed(text)
        return self.model.features(vecs)

    def loss(self, text):
        with torch.no_grad():
            vecs = self.embed(text)
        bs, es = vecs.shape
        vecs1, uvecs = self.model(vecs, undead=True)
        loss0 = self.loss_fn(vecs, vecs1)
        loss1 = self.loss_fn(vecs1 - vecs, uvecs)
        return sqrt(bs * es) * (loss0 + self.undead_weight*loss1)

    def train(
        self, data, epochs=10, max_steps=None, learning_rate=1e-3, epsilon=1e-8, grad_clip=1.0
    ):
        # set up optimizer
        params = self.model.parameters()
        optim = torch.optim.Adam(params, lr=learning_rate, eps=epsilon)

        # evaluate some batches
        def run_eval(ne):
            with torch.no_grad():
                return sum([
                    self.loss(batch).item() for batch in islice(data, ne)
                ]) / ne

        # run some data epochs
        for e in range(epochs):
            # standard gradient update
            for step, batch in enumerate(tqdm(data, desc=f'EPOCH {e}')):
                # accumulate gradients
                optim.zero_grad()
                loss = self.loss(batch)
                loss.backward()

                # process and apply grads
                nn.utils.clip_grad_norm_(params, grad_clip)
                optim.step()

                # break if we hit max steps
                if max_steps is not None and step >= max_steps:
                    break

            # print out loss
            eval_loss = run_eval(8)
            print(f'LOSS: {eval_loss}')
