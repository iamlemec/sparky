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
def dataset_csv(path, columns, batch_size=32):
    datf = pd.read_csv(path, usecols=columns).dropna()[columns]
    text = ['\n'.join(row) for row in datf.itertuples(index=False)]
    return torch.utils.data.DataLoader(text, batch_size=batch_size, shuffle=True)

# average over vector, sum over batch
def mse_loss(x, y, dim=-1):
    return F.mse_loss(x, y, reduction='sum') / x.size(dim)

# here x should be the model and y should be the data
def mse_norm_loss(x, y, dim=-1):
    y_mean = x.mean(dim, keepdim=True).broadcast_to(x.shape)
    return mse_loss(x, y, dim=dim) / mse_loss(y, y_mean, dim=dim)

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
    def __init__(self, embed_size, num_features, topk, pin_bias=True, dead_cutoff=100_000, dead_topk=None, fuzz_factor=1.0):
        super().__init__()
        self.num_features = num_features
        self.topk = topk
        self.dead_cutoff = dead_cutoff
        self.dead_topk = dead_topk
        self.fuzz_factor = fuzz_factor
        self.initialized = False

        self.encoder = Encoder(embed_size, num_features)
        self.decoder = Decoder(embed_size, num_features)

        self.register_buffer('last_usage', torch.zeros(num_features, dtype=torch.int64))
        self.register_buffer('eval_usage', torch.zeros(num_features, dtype=torch.int64))
        self.register_buffer('test_usage', torch.zeros(num_features, dtype=torch.int64))

        if pin_bias:
            self.decoder.bias = self.encoder.bias

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        self.initialized = True

    def save_weights(self, path):
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def init_weights(self, data):
        # initialize bias to data mean
        data_mean = data.mean(dim=0)
        self.encoder.bias.data = data_mean.clone()
        self.decoder.bias.data = data_mean.clone()

        # normalize decoder features
        self.norm_weights()

        # tied intialization of encoder/decoder
        self.decoder.lookup.weight.data = self.encoder.linear.weight.data.clone()

        # set initialized flag
        self.initialized = True

    def norm_weights(self):
        self.decoder.lookup.weight.data /= self.decoder.lookup.weight.data.norm(dim=-1, keepdim=True)

    def reset_usage(self):
        self.eval_usage.fill_(0)

    def features(self, embed):
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1)
        return feats

    # embed: [batch_size, embed_size]
    def forward(self, embed):
        # sparsifry and index
        project = self.encoder(embed)
        weights, feats = project.topk(self.topk, dim=-1, sorted=False)
        embed_recon = self.decoder(feats, weights)

        # accumulate usage stats
        with torch.no_grad():
            batch_size, _ = embed.shape
            total = bincount(feats, self.num_features)
            self.test_usage = total
            self.eval_usage += total
            self.last_usage += batch_size # increment all
            self.last_usage *= 1 - total.clamp(max=1) # zero out used

        # add in resurrected feats
        if self.dead_cutoff is not None:
            # find dead_topk least used features
            dead_mask = self.last_usage > self.dead_cutoff
            dead_total = dead_mask.sum().item()
            if self.dead_topk is None:
                dead_topk = dead_total // 2 # eleuther heuristic
            else:
                dead_topk = min(dead_total, self.dead_topk)

            # reconstruct dead embeddings
            if dead_topk > 0:
                fuzz_sample = self.fuzz_factor * project.std() * torch.randn_like(project)
                dead_project = torch.where(dead_mask, project + fuzz_sample, -torch.inf)
                _, undead_feats = dead_project.topk(dead_topk, dim=-1, sorted=False)
                undead_weights = project.gather(-1, undead_feats)
                undead_recon = self.decoder(undead_feats, undead_weights, bias=False)
            else:
                undead_recon = None
        else:
            undead_recon = None

        # return pair on recons
        return embed_recon, undead_recon

class Trainer:
    def __init__(self, embed, model, learning_rate=1e-4, epsilon=1e-9):
        self.embed = embed
        self.model = model

        # set up optimizer
        self.params = self.model.parameters()
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

    def features(self, text):
        with torch.no_grad():
            vecs = self.embed(text)
        return self.model.features(vecs)

    def loss(self, text, dead_weight=0.0):
        # get embeddings and run model
        with torch.no_grad():
            vecs = self.embed(text)
        vecs1, uvecs = self.model(vecs)
        error = vecs1.detach() - vecs

        # compute loss
        loss0 = mse_norm_loss(vecs1, vecs)
        loss1 = mse_norm_loss(uvecs, error) if uvecs is not None else 0.0
        return loss0 + dead_weight * loss1

    def train(
        self, data, epochs=10, max_steps=None, grad_clip=10.0, eval_steps=8, dead_weight=0.1
    ):
        # intialize weights
        if not self.model.initialized:
            sample_text = next(iter(data))
            sample_vecs = self.embed(sample_text)
            self.model.init_weights(sample_vecs)

        # run some data epochs
        for e in range(epochs):
            # standard gradient update
            prog = tqdm(data, desc=f'EPOCH {e}')
            for step, batch in enumerate(prog):
                # accumulate gradients
                self.optim.zero_grad()
                loss = self.loss(batch, dead_weight=dead_weight)
                loss.backward()

                # process and apply grads
                nn.utils.clip_grad_norm_(self.params, grad_clip)
                self.optim.step()

                # normalize decoder weights
                with torch.no_grad():
                    self.model.norm_weights()

                # print out loss on bar
                test_dead = (self.model.test_usage == 0).float().mean().item()
                prog.set_postfix_str(f'loss = {loss.item():6.4f}, dead = {test_dead:.4f}')

                # save last_usage stats
                torch.save(self.model.last_usage.cpu(), 'last_usage.torch')

                # break if we hit max steps
                if max_steps is not None and step >= max_steps:
                    break

            # print out loss
            with torch.no_grad():
                self.model.reset_usage()
                eval_loss = sum([
                    self.loss(b).item() for b in islice(data, eval_steps)
                ]) / eval_steps
                eval_dead = (self.model.eval_usage == 0).float().mean().item()
            print(f'EVAL: loss = {eval_loss:6.4f}, dead = {eval_dead:.4f}')

            # print out dead feature stats
            usage_mean = self.model.last_usage.float().mean().item()
            dead_feats = (self.model.last_usage > self.model.dead_cutoff).float().mean().item()
            print(f'STAT: last = {usage_mean}, dead = {dead_feats:.4f}')

    def feature_analysis(self, data, num_batches=8):
        # compute feature usage
        usage = torch.zeros((num_batches, self.model.num_features), dtype=torch.float32)
        for i, batch in enumerate(islice(data, num_batches)):
            feats = self.features(batch)
            usage[i, :] = bincount(feats, self.model.num_features)

        # get usage correlation
        used = (usage > 0).float()
        correl = (used @ used.T) / used.sum(dim=1, keepdim=True)

        # get total usage
        total = usage.sum(dim=0) / (num_batches * data.batch_size)

        return total, correl

def setup_trainer(num_features, topk, data_path, embed_path, columns=['title', 'abstract'], batch_size=8192, device='cuda', dtype=torch.float32, **kwargs):
    from ziggy import LlamaCppEmbedding

    # load data
    dataset = dataset_csv(data_path, columns, batch_size=batch_size)

    # load embedding model
    embed = LlamaCppEmbedding(embed_path, dtype=dtype)

    # create model
    sae = AutoEncoder(embed.dims, num_features, topk, **kwargs).to(device=device, dtype=dtype)

    # make trainer
    train = Trainer(embed, sae)

    return train, dataset
