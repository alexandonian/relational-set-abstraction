import itertools
import random
from functools import lru_cache
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cache, cosine_distance


class Relation(torch.nn.Module):
    """Base relation module to model uni-directional relationships.

    A relation maps an ordered set of inputs to a single output representation
    of their uni-directional relationship.

    By convention, the relation is performed on the last two dimensions.

    input[..., num_inputs, in_features] -> output[..., -1, out_features]
    """

    def __init__(self, num_inputs, in_features, out_features, bottleneck_dim=512):
        super(Relation, self).__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck_dim = bottleneck_dim
        self.relate = self.return_mlp()

    def return_mlp(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_inputs * self.in_features, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.out_features),
        )

    def func(self, input):
        out = self.reshape(input)
        return self.relate(out).view(input.size(0), -1, self.out_features)

    def reshape(self, input):
        return input.contiguous().view(-1, self.num_inputs * self.in_features)

    def forward(self, input):
        """Pass concatenated inputs through simple MLP."""
        return self.func(input)


class AbstractionRelationModule(torch.nn.Module):
    """Multi-Relation Abstraction module.

    Args:
        in_features (int):
        out_features (int):
        bottleneck_dim (int):
        scales (int, iterable): Specifies relation scales.

    """

    def __init__(self, in_features, out_features, bottleneck_dim=512, scales=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck_dim = bottleneck_dim

        self.scales = (
            torch.arange(scales, dtype=torch.long) + 1
            if isinstance(scales, int)
            else torch.tensor(scales)
        )
        self.max_scale = max(self.scales)
        self.relations = nn.ModuleList(
            [
                Relation(
                    scale, self.in_features, self.out_features, self.bottleneck_dim
                )
                for scale in self.scales
            ]
        )

        self.linear = nn.Linear(self.max_scale, 1)
        nn.init.constant_(self.linear.weight, 1)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, input):
        n = input.size(-2)
        scales = self.scales[self.scales <= n]
        if len(scales) == 0:
            return self.process_input(input)
        return F.linear(
            torch.stack(
                [
                    torch.stack(
                        [rel(input[..., rset, :]) for rset in self.relation_sets(n, s)]
                    ).mean(0)
                    for s, rel in zip(scales, self.relations)
                ],
                -1,
            ),
            self.linear.weight[..., scales - 1],
            self.linear.bias,
        )[..., 0]

    def process_input(self, input):
        n = input.size(-2)
        min_scale = int(min(self.scales))
        if min_scale // n > 1:
            return self.forward(torch.cat([input for _ in range(min_scale // n)], -2))
        else:
            repeats = random.choice(
                list(itertools.combinations(range(n), min_scale - n))
            )
            return torch.stack(
                [
                    self.forward(torch.cat([input, input[..., rep : rep + 1, :]], -2))
                    for rep in repeats
                ]
            ).mean(0)

    @lru_cache()
    def relation_sets(self, num_inputs, scale):
        return list(itertools.combinations(range(num_inputs), scale))

    @cache
    def name(self):
        return f'SAM_{"-".join([str(s.item()) for s in self.scales])}'


class AbstractionEmbeddingModule(AbstractionRelationModule):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes=391,
        embedding_dim=300,
        bottleneck_dim=512,
        scales=4,
    ):
        super().__init__(in_features, out_features, bottleneck_dim, scales)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.last_linear = nn.Linear(out_features, num_nodes)
        self.embedding_linear = nn.Linear(out_features, embedding_dim)

    def base_forward(self, input):
        out = []
        n = input.size(-2)
        valid_scales = self.scales[self.scales <= n].tolist()
        if len(valid_scales) == 0:
            return self.process_input(input)

        i = 0
        out_dict = {}
        for j, scale in enumerate(range(1, n + 1)):

            # If we have a relation module for the given scale
            if scale in valid_scales:
                rel = self.relations[i]
                i += 1
                scale_out = []
                for rset in self.relation_sets(n, scale):
                    o = rel(input[..., rset, :])
                    out_dict[rset] = o
                    o = torch.stack(
                        [
                            out_val
                            for k, out_val in out_dict.items()
                            if all([kv in rset for kv in k])
                        ]
                    ).sum(dim=0)
                    scale_out.append(o)
                scale_out = torch.cat(scale_out, 1)
                out.append(scale_out)
            else:
                # Obtain scale_out by summing previous scale_outs
                for rset in self.relation_sets(n, scale):
                    if out_dict:
                        o = torch.stack(
                            [
                                out_val
                                for k, out_val in out_dict.items()
                                if all([kv in rset for kv in k])
                            ]
                        ).sum(dim=0)
                        out.append(o)
        out = torch.cat(out, 1)
        return out

    def forward(self, input):
        output = self.base_forward(input)
        embeddings = self.embedding_linear(output)
        node_out = self.last_linear(output).transpose(-2, -1)
        return embeddings, node_out

    def rank(self, input, num_rank=5):
        """Uses learned semantic embeddings."""
        ref_features, rank_features = [self.base_forward(x) for x in input]
        ref_embed = self.embedding_linear(ref_features)[:, -1:]
        rank_embed = self.embedding_linear(rank_features)[:, :num_rank]
        dist = cosine_distance(ref_embed, rank_embed, dim=-1)
        return dist

    def rank_features(self, input, num_rank=5):
        """Uses abstraction features."""
        ref_features, rank_features = [self.base_forward(x) for x in input]
        ref_features = ref_features[:, -1:]
        rank_features = rank_features[:, :num_rank]
        dist = cosine_distance(ref_features, rank_features, dim=-1)
        return dist

    def rank_baseline(self, input, num_rank=5):
        """Uses base model features."""
        ref_features, rank_features = input
        ref_embed = ref_features.mean(dim=1, keepdim=True)
        dist = cosine_distance(ref_embed, rank_features, dim=-1)
        return dist

    def find_outlier(self, input):
        """Uses learned semantic embeddings."""
        num_inputs = input.size(1)
        set_features = self.base_forward(input)
        set_embed = self.embedding_linear(set_features)
        abstr_embed = set_embed[:, num_inputs:].mean(dim=1, keepdim=True)
        class_embed = set_embed[:, :num_inputs]
        dist = cosine_distance(abstr_embed, class_embed, dim=-1)
        return dist, torch.argsort(dist)

    def find_outlier_features(self, input):
        """Uses abstraction features."""
        num_inputs = input.size(1)
        set_features = self.base_forward(input)
        abstr_feats = set_features[:, -1:]
        class_feats = set_features[:, :num_inputs]
        dist = cosine_distance(abstr_feats, class_feats, dim=-1)
        return dist, torch.argsort(dist)

    def find_outlier_baseline(self, input):
        """Uses base model features."""
        set_embed = input.mean(dim=1, keepdim=True)
        dist = cosine_distance(set_embed, input, dim=-1)
        return dist, torch.argsort(dist)

    def find_outlier_rn(self, input):
        num_inputs = input.size(1)
        set_features = self.base_forward(input)
        abstr_feats = set_features[:, -1:]
        dist = cosine_distance(abstr_feats, input, dim=-1)
        return dist, torch.argsort(dist)
        dists = cosine_distance(abstr_feats, set_features, dim=-1)
        pdists = defaultdict(list)
        for i, rset in enumerate(self.relation_sets(num_inputs, min(self.scales))):
            d = dists[:, i]
            for r in rset:
                pdists[r].append(d)

        dist = torch.stack(
            [torch.stack(pdists[n]).sum(dim=0) for n in range(num_inputs)], 1
        )
        return dist, torch.argsort(dist)

    def rank_rn(self, input, num_rank=5):
        num_inputs = input[-1].size(1)
        ref_features, rank_features = [self.base_forward(x) for x in input]
        ref_features = ref_features[:, -1:]
        dists = cosine_distance(ref_features, rank_features, dim=-1)
        pdists = defaultdict(list)
        for i, rset in enumerate(self.relation_sets(num_inputs, min(self.scales))):
            for r in rset:
                pdists[r].append(dists[:, i])

        dist = torch.stack(
            [torch.stack(pdists[n]).sum(dim=0) for n in range(num_inputs)], 1
        )
        return dist
