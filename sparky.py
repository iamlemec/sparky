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
        self.linear = nn.Linear(embed_size, num_features, bias=False)

    def forward(self, embed):
        embed0 = embed - self.bias
        weights = self.linear(embed0)
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
    def __init__(self, embed_size, num_features, topk, pin_bias=True):
        super().__init__()
        self.num_features = num_features
        self.topk = topk

        self.encoder = Encoder(embed_size, num_features)
        self.decoder = Decoder(embed_size, num_features)

        self.register_buffer('last_usage', torch.zeros(num_features, dtype=torch.int64))
        self.register_buffer('eval_usage', torch.zeros(num_features, dtype=torch.int64))

        if pin_bias:
            self.decoder.bias = self.encoder.bias

    def init_weights(self, data):
        # initialize bias to data mean
        data_mean = data.mean(dim=0)
        self.encoder.bias.data = data_mean.clone()
        self.decoder.bias.data = data_mean.clone()

        # normalize decoder features
        self.norm_weights()

        # tied intialization of encoder/decoder
        self.decoder.lookup.weight.data = self.encoder.linear.weight.data.clone()

    def norm_weights(self):
        self.decoder.lookup.weight.data /= self.decoder.lookup.weight.data.norm(dim=-1, keepdim=True)

    def reset_usage(self):
        self.eval_usage.fill_(0)

    def features(self, embed):
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1, sorted=False)
        return feats

    # embed: [batch_size, embed_size]
    def forward(self, embed, dead_cutoff=None, dead_topk=512):
        # sparsifry and index
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1, sorted=False)

        # accumulate usage stats
        with torch.no_grad():
            batch_size, _ = embed.shape
            total = bincount(feats, self.num_features)
            self.eval_usage += total
            self.last_usage += batch_size # increment all
            self.last_usage *= 1 - total.clamp(max=1) # zero out used

        # run decoder
        embed1 = self.decoder(feats, weights)

        # add in resurrected feats
        if dead_cutoff is not None:
            # find dead_topk least used features
            dead_mask = self.last_usage > dead_cutoff
            dead_total = dead_mask.sum().item()
            dead_project = project[:, dead_mask]

            # reconstruct undead embeddings
            undead_topk = np.minimum(dead_topk, dead_total)
            undead_weights, undead_feats = dead_project.topk(undead_topk, dim=-1, sorted=False)
            undead_recon = self.decoder(undead_feats, undead_weights, bias=False)
            return embed1, undead_recon
        else:
            return embed1, None

class Trainer:
    def __init__(self, embed, model):
        self.embed = embed
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def features(self, text):
        with torch.no_grad():
            vecs = self.embed(text)
        return self.model.features(vecs)

    def loss(self, text, dead_cutoff=None, dead_topk=512, dead_weight=0.03):
        # get embeddings and run model
        with torch.no_grad():
            vecs = self.embed(text)
        vecs1, uvecs = self.model(vecs, dead_cutoff=dead_cutoff, dead_topk=dead_topk)

        # compute loss
        scale = sqrt(vecs.numel())
        loss0 = self.loss_fn(vecs, vecs1)
        loss1 = self.loss_fn(vecs1.detach() - vecs, uvecs) if uvecs is not None else 0.0
        return scale * (loss0 + dead_weight*loss1)

    def train(
        self, data, epochs=10, max_steps=None, learning_rate=1e-3, epsilon=1e-8, grad_clip=1.0,
        eval_steps=8, dead_cutoff=10_000, dead_topk=512, dead_weight=0.03, init_weights=True
    ):
        # intialize weights
        if init_weights:
            sample_text = next(iter(data))
            sample_vecs = self.embed(sample_text)
            self.model.init_weights(sample_vecs)

        # set up optimizer
        params = self.model.parameters()
        optim = torch.optim.Adam(params, lr=learning_rate, eps=epsilon)

        # run some data epochs
        for e in range(epochs):
            # standard gradient update
            prog = tqdm(data, desc=f'EPOCH {e}')
            for step, batch in enumerate(prog):
                # accumulate gradients
                optim.zero_grad()
                loss = self.loss(
                    batch, dead_cutoff=dead_cutoff, dead_topk=dead_topk, dead_weight=dead_weight
                )
                loss.backward()

                # process and apply grads
                nn.utils.clip_grad_norm_(params, grad_clip)
                optim.step()

                # normalize decoder weights
                with torch.no_grad():
                    self.model.norm_weights()

                # print out loss on bar
                prog.set_postfix_str(f'LOSS: {loss.item():.4f}')

                # break if we hit max steps
                if max_steps is not None and step >= max_steps:
                    break

            # print out loss
            with torch.no_grad():
                self.model.reset_usage()
                eval_loss = sum([
                    self.loss(b).item() for b in islice(data, eval_steps)
                ]) / eval_steps
                dead_frac = (self.model.eval_usage == 0).float().mean().item()
            print(f'LOSS: {eval_loss}, DEAD: {dead_frac}')

            # print out dead feature stats
            usage_mean = self.model.last_usage.float().mean().item()
            dead_feats = (self.model.last_usage > dead_cutoff).sum().item()
            torch.save(self.model.last_usage.cpu(), 'last_usage.torch')
            print(f'last_usage = {usage_mean}, dead_feats = {dead_feats}')
