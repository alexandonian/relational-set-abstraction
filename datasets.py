import itertools
import json
import pickle
from collections import defaultdict, namedtuple
from typing import Type, Union

import numpy as np
import torch
from torch.utils import data

import cfg
import utils
from graph import AbstractionGraphEmbedding
from utils import cache

DistVideoRecord = namedtuple('DistVideoRecord', ['path', 'label', 'dist'])


class CachedModel(object):
    def __init__(
        self,
        features_file=None,
        list_file=None,
        embedding_file=None,
        format='torch',
        sort_names=False,
        name_suffix='.mp4',
    ):
        self.features_file = features_file
        self.embedding_file = embedding_file
        self.list_file = list_file
        self.format = format
        self.name_suffix = name_suffix
        if embedding_file is not None and ('kinetics' not in embedding_file):
            # Works for moments, does not work for kinetics...
            filenames = (
                sorted(self._embeddings) if sort_names else self._embeddings.keys()
            )
            self.idx2name = {idx: name for idx, name in enumerate(filenames)}
            self.name2idx = {name: idx for idx, name in enumerate(filenames)}
            pass
        # Temp logic to handle differences between moments and kinetics metafiles...
        elif list_file is not None:
            self.name2idx = {}
            self.data_list = []
            self.data_dict = defaultdict(list)
            with open(list_file) as f:
                for i, line in enumerate(f):
                    name, *labels = line.strip().split(' ')
                    labels = [int(lb) for lb in labels]
                    name += self.name_suffix
                    self.name2idx[name] = i
                    self.data_list.append((name, labels))
                    for label in labels:
                        self.data_dict[label].append((name, label))

    def features(self, index):
        return self._features[index]

    def embedding(self, index):
        return self.named_embedding(self.idx2name[index])

    def named_features(self, filename):
        return self.features(self.name2idx[filename])

    def named_embedding(self, filename):
        return self._embeddings[filename]

    @cache
    def _features(self):
        return self.load_data(self.features_file, self.format)

    @cache
    def _embeddings(self):
        return self.load_data(self.embedding_file, self.format)

    def load_data(self, data_file, format):
        if format == 'torch':
            data = torch.load(data_file)
        elif format == 'pickle':
            data = pickle.load(open(data_file, 'rb'))
        else:
            data = np.load(data_file)
            data = {key: data[key].item() for key in data}
        return data

    def __len__(self):
        return len(self._embeddings)


class AltCachedModel(CachedModel):
    def __init__(
        self,
        features_file=None,
        list_file=None,
        embedding_file=None,
        format='torch',
        name_suffix='.mp4',
    ):
        self.features_file = features_file
        self.embedding_file = embedding_file
        self.list_file = list_file
        self.format = format
        self.name_suffix = name_suffix
        self.idx2name = {idx: name for idx, name in enumerate(self._feature_paths)}
        self.name2idx = {name: idx for idx, name in enumerate(self._feature_paths)}

    @cache
    def _feature_data(self):
        return self.load_data(self.features_file, self.format)

    @cache
    def _features(self):
        return self._feature_data['features']

    @cache
    def _feature_paths(self):
        return self._feature_data['path']

    def named_features(self, filename):
        # idx = self._feature_paths.index(filename)
        return self._features[self.name2idx[filename]]

    def named_embedding(self, filename):
        return self._embeddings[filename]

    def embedding(self, index):
        return self.named_embedding(self._feature_paths[index])

    def __len__(self):
        return len(self._features)


def get_cached_abstraction_dataset(name, split, basemodel='resnet3d50'):
    if basemodel in ['i3d']:
        return get_alt_cached_abstraction_dataset(name, split)
    graph_file = cfg.GRAPH_FILE[name]
    list_file = cfg.LIST_FILE[name][split]
    metafile = cfg.METAFILE_FILE[name][split]
    features_file = cfg.FEATURES_FILE[name][split]
    embedding_file = cfg.EMBEDDING_FILE[name][split]
    embed_cat = torch.load(cfg.EMBEDDING_CATEGORIES_FILE[name])
    name_suffix = '' if name in ['moments', 'multimoments'] else '.mp4'
    model = CachedModel(
        features_file=features_file,
        embedding_file=embedding_file,
        list_file=list_file,
        sort_names=(name == 'kinetics'),
        name_suffix=name_suffix,
    )
    record_set = AbstractionSets(metafile)
    graph = AbstractionGraphEmbedding(graph_file, embed_cat)
    return CachedAbstractionDataset(model, record_set, graph)


def get_alt_cached_abstraction_dataset(name, split):
    graph_file = cfg.GRAPH_FILE[name]
    list_file = cfg.LIST_FILE[name][split]
    metafile = cfg.METAFILE_FILE[name][split]
    alt_features_file = cfg.ALT_FEATURES_FILE[name][split]
    embedding_file = cfg.EMBEDDING_FILE[name][split]
    embed_cat = torch.load(cfg.EMBEDDING_CATEGORIES_FILE[name])
    model = AltCachedModel(
        features_file=alt_features_file,
        list_file=list_file,
        embedding_file=embedding_file,
    )
    record_set = AbstractionSets(metafile)
    graph = AbstractionGraphEmbedding(graph_file, embed_cat)
    return CachedAbstractionDataset(model, record_set, graph)


def get_cached_online_abstraction_dataset(name, split):
    graph_file = cfg.GRAPH_FILE[name]
    list_file = cfg.LIST_FILE[name][split]
    features_file = cfg.FEATURES_FILE[name][split]
    embedding_file = cfg.EMBEDDING_FILE[name][split]
    embed_cat = torch.load(cfg.EMBEDDING_CATEGORIES_FILE[name])
    model = CachedModel(
        features_file=features_file,
        embedding_file=embedding_file,
        list_file=list_file,
        sort_names=(name == 'kinetics'),
    )
    graph = AbstractionGraphEmbedding(graph_file, embed_cat)
    return CachedOnlineAbstractionDataset(model, graph)


def get_cached_ranking_dataset(name, split, basemodel='resnet3d50'):
    if basemodel in ['i3d']:
        return
    list_file = cfg.LIST_FILE[name][split]
    graph_file = cfg.GRAPH_FILE[name]
    ranking_file = cfg.RANKING_FILE[name]
    features_file = cfg.FEATURES_FILE[name][split]
    embed_cat = torch.load(cfg.EMBEDDING_CATEGORIES_FILE[name])
    name_suffix = '' if name in ['moments', 'multimoments'] else '.mp4'
    model = CachedModel(
        features_file=features_file,
        list_file=list_file,
        sort_names=(name == 'kinetics'),
        name_suffix=name_suffix,
    )
    record_set = RankingSets(ranking_file)
    graph = AbstractionGraphEmbedding(graph_file, embed_cat)
    return CachedRankingDataset(model, record_set, graph)


def get_cached_outlier_dataset(name, split, basemodel='resnet3d50'):
    if basemodel in ['i3d']:
        return
    list_file = cfg.LIST_FILE[name][split]
    graph_file = cfg.GRAPH_FILE[name]
    ranking_file = cfg.RANKING_FILE[name]
    features_file = cfg.FEATURES_FILE[name][split]
    if split != 'test':
        embedding_file = cfg.EMBEDDING_FILE[name][split]
    else:
        embedding_file = None
    embed_cat = torch.load(cfg.EMBEDDING_CATEGORIES_FILE[name])
    name_suffix = '' if name in ['multimoments', 'moments'] else '.mp4'
    model = CachedModel(
        features_file=features_file,
        embedding_file=embedding_file,
        list_file=list_file,
        sort_names=(name == 'kinetics'),
        name_suffix=name_suffix,
    )
    if split == 'test':
        record_set = RankingSets(ranking_file)
    elif split in ['train', 'val']:
        metafile = cfg.METAFILE_FILE[name][split]
        record_set = AbstractionSets(metafile)
    graph = AbstractionGraphEmbedding(graph_file, embed_cat)
    return CachedOutlierDataset(model, record_set, graph)


def get_cached_abstraction_bce_dataset(name, split, basemodel='resnet3d50'):
    if basemodel in ['i3d']:
        return
    list_file = cfg.LIST_FILE[name][split]
    graph_file = cfg.GRAPH_FILE[name]
    metafile = cfg.METAFILE_FILE[name][split]
    features_file = cfg.FEATURES_FILE[name][split]
    embed_cat = torch.load(cfg.EMBEDDING_CATEGORIES_FILE[name])
    name_suffix = '' if name in ['multimoments', 'moments'] else '.mp4'
    model = CachedModel(
        features_file=features_file,
        list_file=list_file,
        sort_names=(name == 'kinetics'),
        name_suffix=name_suffix,
    )
    record_set = AbstractionSets(metafile)
    graph = AbstractionGraphEmbedding(graph_file, embed_cat)
    return CachedAbstractionBCEDataset(model, record_set, graph)


class AbstractionRecord:
    def __init__(self, data):
        self.data = data

    @property
    def labels_dict(self):
        return {
            tuple(map(int, k.split(','))): v for k, v in self.data['labels'].items()
        }

    @property
    def videos(self):
        return self.data['videos']

    @property
    def labels(self):
        return [self.labels_dict[inds] for inds in self.output_indices()]

    def __len__(self):
        return len(self.videos)

    def output_indices(self, num_inputs=4):
        return [
            rset
            for scale in range(1, len(self) + 1)
            for rset in itertools.combinations(range(num_inputs), scale)
        ]


class AbstractionSets:
    def __init__(self, filename):
        self.records = []
        self.filename = filename
        with open(filename) as f:
            self.data = json.load(f)

        self.records = []
        self.records_dict = defaultdict(list)
        for adata in self.data:
            self.records.append(AbstractionRecord(adata))

    def __getitem__(self, idx: int) -> AbstractionRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class CachedAbstractionDataset(data.Dataset):
    def __init__(
        self,
        model: CachedModel,
        record_set: AbstractionSets,
        graph: Type[AbstractionGraphEmbedding],
    ):
        self.model = model
        self.graph = graph
        self.record_set = record_set

    def __getitem__(self, index):
        try:
            return self._get(index)
        except (IndexError, KeyError, ValueError):
            return self.__getitem__(np.random.randint(len(self)))

    def _get(self, index):
        record = self.record_set[index]
        num_videos = len(record.videos)

        categories = [cat for cat_list in record.labels for cat in cat_list]
        categories = self.graph.category_to_node(*categories)
        labels = torch.LongTensor(self.graph.node_to_label(*categories))

        features = torch.stack(
            [self.model.named_features(name) for name in record.videos]
        )
        class_embeddings = torch.stack(
            [self.model.named_embedding(name) for name in record.videos]
        )
        abstr_embeddings = self.graph.node_to_embedding(*categories[num_videos:])
        embeddings = torch.cat((class_embeddings, abstr_embeddings))
        return features, (embeddings, labels)

    def __len__(self):
        return len(self.record_set)


class CachedAbstractionBCEDataset(CachedAbstractionDataset):
    def __init__(
        self,
        model: CachedModel,
        record_set: AbstractionSets,
        graph: Type[AbstractionGraphEmbedding],
    ):
        self.model = model
        self.graph = graph
        self.record_set = record_set

    def __getitem__(self, index):
        try:
            return self._get(index)
        except (IndexError, KeyError):
            return self.__getitem__(np.random.randint(len(self)))

    def _get(self, index):
        record = self.record_set[index]
        categories = [cat for cat_list in record.labels for cat in cat_list]
        nodes = self.graph.category_to_node(*categories)
        labels = []
        for n in nodes:
            l = torch.LongTensor(self.graph.hypernym_path_labels(n)).t()
            size = (1, self.graph.num_nodes)
            ll = utils.encode_one_hot(size, l)
            labels.append(ll)
        labels = torch.cat(labels)

        features = torch.stack(
            [self.model.named_features(name) for name in record.videos]
        )
        return features, (labels,)


class CachedOnlineAbstractionDataset(data.Dataset):
    def __init__(
        self,
        model: CachedModel,
        graph: Type[AbstractionGraphEmbedding],
        num_inputs: int = 4,
        name_suffix: str = '',
    ):
        self.model = model
        self.graph = graph
        self.num_inputs = num_inputs
        self.name_suffix = name_suffix

    def __getitem__(self, index):
        try:
            return self._get(index)
        except IndexError:
            return self.__getitem__(np.random.randint(len(self)))

    def descendant_instances(self, node):
        labels = self.graph.node_to_label(*self.graph.descendants(node))
        instances = itertools.chain.from_iterable(
            [self.model.data_dict[lab] for lab in labels]
        )
        return instances

    def _get(self, index):
        name, labels = self.model.data_list[index]
        node = np.random.choice(self.graph.label_to_node(*labels))
        ancestors = list(self.graph.ancestors(node))
        ancestor = np.random.choice(ancestors)
        successors = list(self.descendant_instances(ancestor))
        np.random.shuffle(successors)

        names, labels = zip(
            *[
                successors[np.random.choice(range(len(successors)))]
                for _ in range(self.num_inputs - 1)
            ]
        )
        names = (name,) + names
        labels = tuple(self.graph.node_to_label(node)) + labels
        nodes = self.graph.label_to_node(*labels)
        abstr_nodes = []
        for scale in range(1, self.num_inputs + 1):
            for node_set in itertools.combinations(nodes, scale):
                abstr_node = self.graph.lowest_common_ancestor(*node_set)
                abstr_nodes.append(np.random.choice(abstr_node))

        labels = self.graph.node_to_label(*abstr_nodes)
        labels = torch.LongTensor(labels)

        features = torch.stack(
            [self.model.named_features(name + self.name_suffix) for name in names]
        )
        class_embeddings = torch.stack(
            [self.model.named_embedding(name + '.mp4') for name in names]
        )
        abstr_embeddings = self.graph.node_to_embedding(*abstr_nodes[self.num_inputs :])
        embeddings = torch.cat((class_embeddings, abstr_embeddings))
        return features, (embeddings, labels)

    def __len__(self):
        return len(self.model)


class RankingRecord:
    """A single reference-query record."""

    def __init__(self, data):
        self.data = data

    @property
    def common(self):
        return self.data['common']

    @property
    def query(self):
        return [DistVideoRecord(*x) for x in self.data['query']]

    @property
    def reference(self):
        return [DistVideoRecord(*x) for x in self.data['reference']]

    @property
    def query_size(self):
        return len(self.query)

    @property
    def reference_size(self):
        return len(self.reference)

    @property
    def ranking(self):
        return np.argsort(self.dists)

    @property
    def outlier_idx(self):
        return np.argmax(self.dists)

    @property
    def outlier(self):
        return self.query[self.outlier_idx]

    @property
    def dists(self):
        return [r.dist for r in self.query]


class RankingSets:
    """Encapsulates a collection of reference-query sets."""

    def __init__(self, filename):
        self.records = []
        self.filename = filename
        with open(filename) as f:
            self.data = json.load(f)

        self.records = []
        self.records_dict = defaultdict(list)
        for size, rank_data in self.data.items():
            for d in rank_data:
                self.records.append(RankingRecord(d))
                self.records_dict[size].append(RankingRecord(d))

    def __getitem__(self, idx: int) -> RankingRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class CachedRankingDataset(data.Dataset):
    def __init__(
        self,
        model: CachedModel,
        record_set: RankingSets,
        graph: Type[AbstractionGraphEmbedding],
    ):
        self.model = model
        self.graph = graph
        self.record_set = record_set

    def get_dist(self, index):
        return self.record_set[index].dists

    def get_dists(self):
        return np.stack([self.get_dist(i) for i in range(len(self))])

    def get_ranking(self, index):
        return self.record_set[index].ranking

    def get_rankings(self):
        return np.stack([self.get_ranking(i) for i in range(len(self))])

    def __getitem__(self, index: int):
        record = self.record_set[index]

        ref_features = torch.stack(
            [
                self.model.named_features(rec.path.replace('multi_test', 'test_multi'))
                for rec in record.reference
            ]
        )
        rank_features = torch.stack(
            [
                self.model.named_features(rec.path.replace('multi_test', 'test_multi'))
                for rec in record.query
            ]
        )
        return ref_features, rank_features

    def __len__(self):
        return len(self.record_set)


class CachedOutlierDataset(data.Dataset):
    def __init__(
        self,
        model: CachedModel,
        record_set: Union[AbstractionSets, RankingSets],
        graph: Type[AbstractionGraphEmbedding],
    ):
        self.model = model
        self.graph = graph
        if isinstance(record_set, RankingSets):
            record_set.records = [r for r in record_set.records if len(r.reference) > 1]
        elif isinstance(record_set, AbstractionSets):
            record_set.records = [r for r in record_set.records if len(r.videos) > 2]
        self.record_set = record_set

    def get_dist(self, index):
        return self.record_set[index].dists

    def get_dists(self):
        return np.stack([self.get_dist(i) for i in range(len(self))])

    def get_ranking(self, index):
        return self.record_set[index].ranking

    def get_rankings(self):
        return np.stack([self.get_ranking(i) for i in range(len(self))])

    def __getitem__(self, index):
        try:
            return self._get(index)
        except (IndexError, KeyError):
            return self.__getitem__(np.random.randint(len(self)))

    def _get(self, index: int):

        if isinstance(self.record_set, RankingSets):
            record = self.record_set[index]

            # Outlier is last element in the set.
            set_names = record.reference + [record.outlier]
            features = torch.stack(
                [
                    self.model.named_features(
                        rec.path.replace('multi_test', 'test_multi')
                    )
                    for rec in set_names
                ]
            )
            return features, torch.LongTensor([len(set_names) - 1])

        elif isinstance(self.record_set, AbstractionSets):
            record = self.record_set[index]
            outliers = []

            features = torch.stack(
                [self.model.named_features(name) for name in record.videos]
            )
            class_embeddings = torch.stack(
                [self.model.named_embedding(name) for name in record.videos]
            )
            inds = [ind for ind in record.output_indices() if len(ind) > 2]
            for ind in inds:
                label = record.labels_dict[ind]
                nodes = self.graph.category_to_node(*label)
                ce = torch.stack([class_embeddings[i] for i in ind])
                abstr_embedding = self.graph.node_to_embedding(*nodes)
                dists = utils.cosine_distance(abstr_embedding, ce, dim=-1)
                outlier_idx = ind[torch.argmax(dists)]
                outliers.append(outlier_idx)

            return features, (torch.LongTensor(outliers),)

    def __len__(self):
        return len(self.record_set)
