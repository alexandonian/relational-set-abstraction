import itertools
import json
import random
from collections import defaultdict
from functools import lru_cache

import networkx as nx
import numpy as np
import torch

from utils import cache


class AbstractionGraph(nx.DiGraph):
    def __init__(self, data=None, **attr):
        if isinstance(data, str):
            data = self.from_jsonfile(data)
        super().__init__(data, **attr)
        self.graph['node_to_label'] = {n: nd['label'] for n, nd in self.nodes.data()}
        self.graph['label_to_node'] = {nd['label']: n for n, nd in self.nodes.data()}
        self.graph['category_to_node'] = {
            nd.get('category', n): n for n, nd in self.nodes.data()
        }
        self._n2l = self.graph['node_to_label']
        self._l2n = self.graph['label_to_node']
        self._c2n = self.graph['category_to_node']
        self.seed = attr.get('seed', 0)
        random.seed(self.seed)

    def to_json(self):
        return nx.readwrite.json_graph.node_link_data(self)

    def to_jsonfile(self, jsonfile):
        data = self.to_json()
        json.dump(data, open(jsonfile, 'w'))

    @classmethod
    def from_json(cls, data):
        return cls(nx.readwrite.json_graph.node_link_graph(data))

    @classmethod
    def from_jsonfile(cls, jsonfile):
        return cls.from_json(json.load(open(jsonfile)))

    def descendants(self, *nodes, common=True, include_src=True):
        f = set.intersection if common else set.union
        dsc = (
            (lambda s, n: nx.descendants(s, n).union({n}))
            if include_src
            else nx.descendants
        )
        return f(*[dsc(self, n) for n in nodes])

    def ancestors(self, *nodes, common=True, include_src=True):
        f = set.intersection if common else set.union
        anc = (
            (lambda s, n: nx.ancestors(s, n).union({n}))
            if include_src
            else nx.ancestors
        )
        return f(*[anc(self, n) for n in nodes])

    def hypernym_path(self, node, root='root'):
        return list(nx.shortest_path(self, root, node))[::-1]

    def hypernym_paths(self, *nodes, root='root'):
        return [self.hypernym_path(n, root=root) for n in nodes]

    def hypernym_path_labels(self, node, root='root'):
        return list(map(self.node_to_label, self.hypernym_path(node, root)))

    def hypernym_paths_labels(self, *nodes, root='root'):
        return [self.hypernym_path_labels(n, root) for n in nodes]

    @cache
    def undirected(self):
        return nx.Graph(self)

    @lru_cache()
    def node_distance(self, source, target, undirected=True):
        if source == target:
            return 0
        G = self.undirected if undirected else self
        return nx.shortest_path_length(G, source, target)

    def path_length(self, *nbunch, undirected=True):
        return [self.node_distance(s, t, undirected) for s, t in nbunch]

    def node_depth(self, node, root='root'):
        return self.node_distance(root, node, undirected=False)

    def lowest_common_ancestor(self, *nodes, root='root', include_src=True):
        """Find the lowest common ancestor in the directed, acyclic graph of *nodes.

        Notes:
            This definition is the opposite of the term as it is used e.g. in biology!

        Arguments:
            *nodes (list): Node IDs in the DAG.
            root (node): Root node ID (default: 'root')

        Returns:
            list: [node 1, ..., node n]
                list of lowest common ancestor nodes (can be more than one)

        """
        assert nx.is_directed_acyclic_graph(
            self
        ), "Graph has to be acyclic and directed."

        # Get ancestors of both (intersection)
        common_ancestors = list(
            self.ancestors(*nodes, common=True, include_src=include_src)
        )

        # Get sum of path lengths
        sum_path_len = np.zeros((len(common_ancestors)))
        for i, c in enumerate(common_ancestors):
            sum_path_len[i] = sum(nx.shortest_path_length(self, c, n) for n in nodes)

        # Return minima
        try:
            (minima,) = np.where(sum_path_len == np.min(sum_path_len))
        except ValueError:
            return [root]
        else:
            return [common_ancestors[i] for i in minima]

    def set_abstraction(self, *nodes, root='root'):
        return self.lowest_common_ancestor(*nodes, root=root)

    def label_to_node(self, *labels):
        return [self._l2n[l] for l in labels]

    def category_to_node(self, *categories):
        # return [self._c2n[c] for c in categories]
        return [self._c2n.get(c, c) for c in categories]

    def node_to_label(self, *nodes):
        return [self._n2l[n] for n in nodes]

    def determine_outlier(self, *nodes):
        abstr = defaultdict(list)
        for node1, node2 in itertools.combinations(nodes, 2):
            a = self.lowest_common_ancestor(node1, node2)
            abstr[node1] += a
            abstr[node2] += a
        abstr = {n: set(a) for n, a in abstr.items()}
        a = list(abstr.values())
        if a.count(a[0]) == len(a):
            return -1
        num_common_anc = np.array([len(abstr[n]) for n in nodes])
        try:
            (minima,) = np.where(num_common_anc == np.min(num_common_anc))
        except ValueError:
            return -1
        else:
            return int(minima[0])

    def is_concrete(self, *nodes):
        return [l < self.num_class for l in self.node_to_label(*nodes)]

    def is_abstract(self, *nodes):
        def inv(x):
            return not x

        return list(map(inv, self.is_concrete(*nodes)))

    def is_leaf(self, *nodes):
        return [len(self.descendants(n)) == 0 for n in nodes]

    def abstraction_sets(self, node, num_inputs=2):
        sc_dc = {s: self.descendants(s) for s in list(self.successors(node))}
        descs = [d if d else {s} for s, d in sc_dc.items()]
        return list(
            itertools.chain(
                *[
                    list(itertools.product(*x))
                    for x in itertools.permutations(descs, num_inputs)
                ]
            )
        )

    @cache
    def num_class(self):
        return self.graph['num_class']

    @cache
    def num_abstract_nodes(self):
        return self.num_nodes - self.num_class

    @cache
    def concrete_nodes(self):
        return [n for n in self.nodes if self.is_concrete(n)[0]]

    @cache
    def abstract_nodes(self):
        return [n for n in self.nodes if self.is_abstract(n)[0]]

    @cache
    def leaf_nodes(self):
        return [n for n in self.nodes if self.is_leaf(n)[0]]

    @cache
    def num_nodes(self):
        return self.graph['num_nodes']

    @cache
    def max_depth(self):
        return nx.dag_longest_path_length(self)

    @property
    def max_ancestors(self):
        return max([len(self.ancestors(n)) for n in self.nodes])

    @property
    def max_descendants(self):
        return max([len(self.descendants(n)) for n in self.nodes])

    @property
    def max_predecessors(self):
        return max([len(list(self.predecessors(n))) for n in self.nodes])

    @property
    def max_successors(self):
        return max([len(list(self.successors(n))) for n in self.nodes])

    @staticmethod
    def swap_nodes(G, *ebunch):
        for u, v in ebunch:
            u_pred = list(G.predecessors(u))
            v_pred = list(G.predecessors(v))
            u_succ = list(G.successors(u))
            v_succ = list(G.successors(v))
            if u in v_pred and v in u_succ:
                u_succ += u
            elif v in u_pred and u in v_succ:
                v_succ += v

            # Remove old u Parents + Children
            G.remove_edges_from([(p, u) for p in u_pred])
            G.remove_edges_from([(u, s) for s in u_succ])

            # Remove old v Parents + Children
            G.remove_edges_from([(p, v) for p in v_pred])
            G.remove_edges_from([(v, s) for s in v_succ])

            # Add v Parents + Children to u
            G.add_edges_from([(p, u) for p in v_pred])
            G.add_edges_from([(u, s) for s in v_succ])

            # Add u Parents + Children to v
            G.add_edges_from([(p, v) for p in u_pred])
            G.add_edges_from([(v, s) for s in u_succ])
        return G

    def permute_concrete_nodes(self):
        self.permute_nodes(*self.concrete_nodes)

    def permute_abstract_nodes(self, keeproot=True):
        self.permute_nodes(*self.abstract_nodes, keeproot=keeproot)

    def permute_all_nodes(self, keeproot=True):
        self.permute_nodes(*self.nodes, keeproot=keeproot)

    def permute_nodes(self, *nodes, keeproot=True):
        nodes = list(nodes)
        random.seed(self.seed)
        random.shuffle(nodes)
        half = len(nodes) // 2
        swaps = list(zip(*[nodes[:half], nodes[half:]]))
        if keeproot:
            swaps.extend([x[::-1] for x in swaps if 'root' in x])
        self = self.swap_nodes(self, *swaps)
        return self

    def _remove_node(self, node):
        print(f'removing: {node}')
        successors = list(self.successors(node))
        predecessors = list(self.predecessors(node))
        self.add_edges_from(
            [(pred, succ) for pred in predecessors for succ in successors]
        )
        self.remove_node(node)

    def remove_nodes(self, num_drop, class_only=True, drop_complement=False):
        random.seed(self.seed)
        limit = self.num_class if class_only else len(self)
        exclude = random.sample(range(limit), num_drop)
        if drop_complement:
            exclude = [node for node in range(limit) if node not in exclude]
        node_remove = [n for n in self if self.node[n]['label'] in exclude]
        [self._remove_node(node) for node in node_remove]
        # node_remove.extend(self._refine_graph())
        self.graph['removed_nodes'] = node_remove
        self.graph['removed_labels'] = exclude
        return node_remove, exclude

    def _refine_graph(self):
        cleanup = []
        for n in self.node:
            succ = list(self.successors(n))
            if len(succ) == 1 and (self.node[n].get('category', None) is None):
                print(f'also cleaning up: {n}')
                cleanup.append(n)
        [self._remove_node(node) for node in cleanup]
        print(f'num cleanup: {len(cleanup)}')
        return cleanup

    def to_node(method):
        # TODO: Decorate methods so that they accept nodes or labels
        def _func(self, *args, **kwargs):
            is_node = [x in self for x in args]
            node = [x if x in self else self._l2n[x] for x in args]
            out = method(self, *node, **kwargs)
            out = [
                self.node_to_label(x) if is_n else x for x, is_n in zip(out, is_node)
            ]
            return out

        return _func

    def permute_graph(self, class_only=True):
        limit = self.num_class if class_only else len(self)
        include = list(range(limit))
        nodes_permute = [n for n in self if self.node[n]['label'] in include]
        self = self.permute_nodes(*nodes_permute)


class AbstractionGraphEmbedding(AbstractionGraph):
    def __init__(self, data=None, embedding=None, **attr):
        super().__init__(data, **attr)
        self.embedding = embedding
        self.graph['node_to_name'] = {node: node for node in self.nodes}
        self.graph['label_to_node'] = {nd['label']: n for n, nd in self.nodes.data()}
        self._n2l = self.graph['node_to_label']
        self._l2n = self.graph['label_to_node']
        self._node2name = self.graph['node_to_name']
        self.seed = attr.get('seed', 0)

    def node_to_name(self, *nodes):
        return [self._node2name[n] for n in nodes]

    def node_to_embedding(self, *nodes):
        return torch.stack([self.embedding[name] for name in self.node_to_name(*nodes)])

    def label_to_embedding(self, *labels):
        return self.node_to_embedding(*self.label_to_node(*labels))


class AbstractionGraphFastTextEmbedding(AbstractionGraph):
    def __init__(self, data=None, embedding=None, **attr):
        super().__init__(data, **attr)
        self.embedding = embedding
        self.graph['node_to_name'] = {}
        # for n, nd in self.nodes.data():
        #     if n == 'root':
        #         continue
        #     try:
        #         # name = wn.synset(n).lemmas()[0].name() + 'ing'
        #         name = nd['category']
        #     except KeyError:
        #         name = wn.synset(n).lemmas()[0].name()
        #     self.graph['node_to_name'][n] = name
        self.graph['label_to_node'] = {nd['label']: n for n, nd in self.nodes.data()}
        self._n2l = self.graph['node_to_label']
        self._l2n = self.graph['label_to_node']
        # self._node2name = self.graph['node_to_name']
        self.seed = attr.get('seed', 0)

    def node_to_name(self, *nodes):
        return [self._node2name[n] for n in nodes]

    def node_to_embedding(self, *nodes):
        return torch.stack(
            [
                torch.from_numpy(self.embedding[name])
                for name in self.node_to_name(*nodes)
            ]
        )

    def label_to_embedding(self, *labels):
        return self.node_to_embedding(*self.label_to_node(*labels))


class AbstractionGraphModule(AbstractionGraph):
    def __init__(self, data=None, permute_labels=False, **attr):
        super().__init__(data, **attr)
        if permute_labels:
            print('Permuting concrete nodes')
            self.permute_graph()
        self.ancestor_tensors = self._ancestor_tensors()
        self.descendant_tensors = self._descendant_tensors()
        self.predecessor_tensors = self._predecessor_tensors()
        self.successor_tensors = self._successor_tensors()

    @cache
    def abstr_inds(self):
        return torch.LongTensor(self.node_to_label(*self.abstract_nodes))

    @cache
    def class_inds(self):
        return torch.LongTensor(self.node_to_label(*self.concrete_nodes))

    @cache
    def leaf_inds(self):
        return torch.LongTensor(self.node_to_label(*self.leaf_nodes))

    def _ancestor_tensors(self):
        ancs_mask = torch.zeros(self.num_class, self.max_ancestors, 1)
        ancs_inds = torch.zeros(self.num_class, self.max_ancestors).long()
        for i in range(self.num_class):
            node = self.label_to_node(i)
            ancs = list(self.ancestors(*node))
            labels, n = self.node_to_label(*ancs), len(ancs)
            ancs_mask[i, :n] = torch.ones(n, 1)
            ancs_inds[i, :n] = torch.LongTensor(labels)
        return [ancs_inds, ancs_mask, ancs_mask.sum(-2)]

    def _descendant_tensors(self):
        desc_mask = torch.zeros(self.num_nodes, self.max_descendants, 1)
        desc_inds = torch.zeros(self.num_nodes, self.max_descendants).long()
        for i in range(self.num_nodes):
            node = self.label_to_node(i)
            desc = list(self.descendants(*node))
            desc = [d for d in desc if d in self.concrete_nodes]
            labels, n = self.node_to_label(*desc), len(desc)
            desc_mask[i, :n] = torch.ones(n, 1)
            desc_inds[i, :n] = torch.LongTensor(labels)
        num_desc = desc_mask.sum(-2)
        num_desc[num_desc == 0] = 1
        return [desc_inds, desc_mask, num_desc]

    def _predecessor_tensors(self):
        pred_mask = torch.zeros(self.num_class, self.max_predecessors, 1)
        pred_inds = torch.zeros(self.num_class, self.max_predecessors).long()
        for i in range(self.num_class):
            node = self.label_to_node(i)
            ancs = list(self.predecessors(*node))
            labels, n = self.node_to_label(*ancs), len(ancs)
            pred_mask[i, :n] = torch.ones(n, 1)
            pred_inds[i, :n] = torch.LongTensor(labels)
        num_pred = pred_mask.sum(-2)
        num_pred[num_pred == 0] = 1
        return [pred_inds, pred_mask, num_pred]

    def _successor_tensors(self):
        succ_mask = torch.zeros(self.num_nodes, self.max_successors, 1)
        succ_inds = torch.zeros(self.num_nodes, self.max_successors).long()
        for i in range(self.num_nodes):
            node = self.label_to_node(i)
            succ = list(self.successors(*node))
            labels, n = self.node_to_label(*succ), len(succ)
            succ_mask[i, :n] = torch.ones(n, 1)
            succ_inds[i, :n] = torch.LongTensor(labels)
        num_succ = succ_mask.sum(-2)
        num_succ[num_succ == 0] = 1
        return [succ_inds, succ_mask, num_succ]

    def update_abstr(self, class_out, abstr_out):
        """Update abstr_out using class_out."""
        inds, mask, num = [x.to(abstr_out.device) for x in self.descendant_tensors]
        return ((abstr_out[:, inds] * mask).sum(-2) / num).mean(-1, keepdim=True)

    def update_class(self, class_out, abstr_out):
        """Update class_out using abstr_out."""
        inds, mask, num = [x.to(abstr_out.device) for x in self.ancestor_tensors]
        return class_out + ((abstr_out[:, inds] * mask).sum(-2) / num)


class DiGraphSynTextNode(nx.DiGraph):
    def __init__(self, data=None, **attr):
        super(DiGraphSynTextNode, self).__init__(incoming_graph_data=data, **attr)
        self.root = data

    @property
    def root(self):
        if not self._root.is_root:
            self._root = self._root.root
            return self.root
        else:
            return self._root

    @root.setter
    def root(self, value):
        self._root = value

    def remove_node_obj(self):
        for node in self.node:
            try:
                self.node[node].pop('node')
            except KeyError:
                pass

    def extend_hypernymns(self):
        edges = []
        for node in self.node:
            node = self.node[node]['node']
            edges.extend([(parent, node) for parent in node.hypernyms()])
        self.add_edges_from([(parent.name(), child.name()) for parent, child in edges])
        for parent, child in edges:
            self.node[parent.name()]['node'] = parent
            self.node[child.name()]['node'] = child

    def extend_hypernymn_paths(self):
        edges = []
        for node in self.node:
            try:
                node = self.node[node]['node']
                for nodes in node.hypernym_paths():
                    pairs = [(parent, child) for parent, child in zip(nodes, nodes[1:])]
                    edges.extend([(parent, child) for parent, child in pairs])
            except KeyError:
                pass
        self.add_edges_from([(parent.name(), child.name()) for parent, child in edges])
        for parent, child in edges:
            self.node[parent.name()]['node'] = parent
            self.node[child.name()]['node'] = child


def collapse_graph(G, node):
    try:
        successors = list(G.successors(node))
        predecessors = list(G.predecessors(node))
        if len(successors) == 1 and len(predecessors) == 1:
            if 'label' not in G.node[node]:
                G.add_edge(predecessors[0], successors[0])
                G.remove_node(node)
                collapse_graph(G, successors[0])
        elif len(successors) > 0:
            for s in successors:
                collapse_graph(G, s)
    except Exception as e:
        print(e)


def make_torch_embedding(embedding):
    def get_word_vector(self, word):
        vec = self.get_word_vector(word)
        return torch.from_numpy(vec)

    setattr(embedding.__class__, 'get_word_vector', get_word_vector)
    return embedding
