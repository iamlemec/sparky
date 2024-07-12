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
    def __init__(self, embed_size, num_features, topk, pin_bias=True):
        super().__init__()
        self.num_features = num_features
        self.topk = topk

        self.encoder = Encoder(embed_size, num_features)
        self.decoder = Decoder(embed_size, num_features)

        self.register_buffer('last_usage', torch.zeros(num_features, dtype=torch.int64))
        self.register_buffer('eval_usage', torch.zeros(num_features, dtype=torch.int64))

        self.init_weights()
        if pin_bias:
            self.decoder.bias = self.encoder.bias

    def init_weights(self):
        self.encoder.bias.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.encoder.weight.data.normal_(0, 0.1)
        self.decoder.lookup.weight.data.normal_(0, 0.1)

    def reset_usage(self):
        self.eval_usage.fill_(0)

    def features(self, embed):
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1, sorted=False)
        return feats

    # embed: [batch_size, embed_size]
    def forward(self, embed, random_samp=None):
        # sparsifry and index
        project = self.encoder(embed)
        weights0, feats0 = project.topk(self.topk, dim=-1, sorted=False)

        # randomize some of the features
        if random_samp is not None:
            batch_size, _ = embed.shape
            random_feats = torch.randint(self.num_features, (batch_size, random_samp), device=embed.device)
            random_weights = torch.gather(project, -1, random_feats)
            feats = torch.cat([feats0, random_feats], dim=-1)
            weights = torch.cat([weights0, random_weights], dim=-1)
        else:
            feats = feats0
            weights = weights0

        # accumulate usage stats
        with torch.no_grad():
            batch_size, _ = embed.shape
            total = bincount(feats0, self.num_features)
            self.eval_usage += total
            self.last_usage += batch_size # increment all
            self.last_usage *= 1 - total.clamp(max=1) # zero out used

        # run decoder
        recon = self.decoder(feats, weights)
        embed1 = F.normalize(recon)

        # compute the entropy of the distribution
        logits = weights - torch.logsumexp(weights, dim=-1, keepdim=True)
        entropy = -(logits.exp()*logits).sum(dim=-1)

        return embed1, entropy

class Trainer:
    def __init__(self, embed, model):
        self.embed = embed
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def features(self, text):
        with torch.no_grad():
            vecs = self.embed(text)
        return self.model.features(vecs)

    def loss(self, text, entropy_weight=None, random_samp=None):
        # get embeddings and run model
        with torch.no_grad():
            vecs = self.embed(text)
        vecs1, entropy = self.model(vecs, random_samp=random_samp)

        # compute loss
        bs, es = vecs.shape
        loss_mse = self.loss_fn(vecs, vecs1)
        loss_ent = entropy_weight * entropy.mean() if entropy_weight is not None else 0.0
        return sqrt(bs) * ( sqrt(es) * loss_mse - loss_ent )

    def train(
        self, data, epochs=10, max_steps=None, learning_rate=1e-3, epsilon=1e-8, grad_clip=1.0,
        eval_steps=8, entropy_weight=0.01, random_samp=32, dead_cutoff=100_000
    ):
        # set up optimizer
        params = self.model.parameters()
        optim = torch.optim.Adam(params, lr=learning_rate, eps=epsilon)

        # run some data epochs
        for e in range(epochs):
            # standard gradient update
            for step, batch in enumerate(tqdm(data, desc=f'EPOCH {e}')):
                # accumulate gradients
                optim.zero_grad()
                loss = self.loss(batch, entropy_weight=entropy_weight, random_samp=random_samp)
                loss.backward()

                # process and apply grads
                nn.utils.clip_grad_norm_(params, grad_clip)
                optim.step()

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
            print(f'last_usage = {usage_mean}, dead_feats = {dead_feats}')
