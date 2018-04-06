# -*- coding: utf-8 -*-
#
#  Copyright 2018 Pascual Martinez-Gomez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from collections import Counter
from collections import defaultdict
import itertools
import logging
import numpy as np

from nltk2graph import formula_to_graph
from nltk2graph import get_label
from nltk2graph import get_node_token
from nltk2graph import make_empty_graph

import networkx as nx

class GraphStructures(object):
    """
    For a certain graph, it indexes graph structures for all its nodes.
    """

    def __init__(self, graph):
        self.graph = nx.convert_node_labels_to_integers(graph, first_label=1)
        # Child nodes.
        self.children = defaultdict(list)
        # Parent nodes.
        self.parents = defaultdict(list)
        # Treelets where the current node participates as the predicate.
        self.treelets_predicate = defaultdict(list)
        # Treelets where the current node participates as the left child.
        self.treelets_left = defaultdict(list)
        # Treelets where the current node participates as the right child.
        self.treelets_right = defaultdict(list)
        self.collect_structures()
        return

    def collect_structures(self):
        """
        It populates the structure dictionaries.
        """

        # Get children and parent relations.
        for src, trg in self.graph.edges:
            self.children[src].append(trg)
            self.parents[trg].append(src)

        # Get treelet relations.
        for nid in self.graph.nodes:
            if get_label(self.graph, nid, 'type') == 'constant':
                succs = list(self.graph.successors(nid))
                succs.sort(key=lambda x: get_label(self.graph, x, 'arg', 0))
                combs = itertools.combinations(succs, 2)
                for left, right in combs:
                    self.treelets_predicate[nid].append((left, right))
                    self.treelets_left[left].append((nid, right))
                    self.treelets_right[right].append((left, nid))
        return

# TODO: Set defaults.
# TODO: Copy parameters.
# TODO: Manage data splitting.
# TODO: Make all parameters explicit in __init__.
# TODO: set maximum number of words and substitute other occurrences by <unk>.
# TODO: treat <unk> differently to padding.
class GraphData(object):
    """
    Manages multiple graphs and transforms them into matrices
    for deep learning.
    """

    def __init__(self,
        graph_structs,
        max_nodes=None,
        max_bi_relations=None,
        max_tri_relations=None):

        self.graph_structs = graph_structs

        self._max_nodes = max_nodes
        self._max_bi_relations = max_bi_relations
        self._max_treelets = max_tri_relations
        self.emb_dim = None

        self.word2ind = defaultdict(lambda: len(self.word2ind))
        self.special_tokens = [
            '<unk>', '<exists>', '<all>', '<&>', '<|>',
            '<=>', '<Subj>', '<root>']
        self.word2ind['<unk>'] # index 0 for unknown word.

        # One big matrix with node embeddings for all graphs.
        self.node_embs = None
        # For each graph, specifies the global node indices: |graphs| x max_nodes
        self.node_inds = None

        # Node relationships. `children` and `parents` are binary relations.
        self.children = None
        self.parents = None
        # These are ternary relations.
        self.treelets_predicate = None
        self.treelets_right = None
        self.treelets_left = None

        # Normalizers.
        self.birel_child_norm = None
        self.birel_parent_norm = None
        self.treelets_norm = None

    def copy_parameters(self, graph_data):
        assert isinstance(graph_data, self.__class__)
        self._max_nodes = graph_data._max_nodes
        self._max_bi_relations = graph_data._max_bi_relations
        self._max_treelets = graph_data._max_treelets
        self.word2ind = graph_data.word2ind
        self.node_embs = graph_data.node_embs

    @staticmethod
    def from_formulas(formulas,
        max_nodes=None,
        max_bi_relations=None,
        max_tri_relations=None,
        emb_dim=128):
        graphs = []
        for formula in formulas:
            try:
                graph = formula_to_graph(formula, normalize=True)
            except Exception as e:
                graph = make_empty_graph()
            graphs.append(graph)
        graph_structs = [GraphStructures(g) for g in graphs]
        graph_data = GraphData(graph_structs, max_bi_relations, max_tri_relations)
        graph_data.emb_dim = emb_dim
        return graph_data

    def get_max_nodes(self):
        if self._max_nodes is not None:
            return self._max_nodes
        self._max_nodes = max(len(gs.graph.nodes) for gs in self.graph_structs) + 1
        return self._max_nodes

    def get_num_words(self):
        return len(self.word2ind)

    def get_max_bi_relations(self):
        if self._max_bi_relations is not None:
            return self._max_bi_relations
        max_children = max(
            len(chs) for gs in self.graph_structs for chs in gs.children.values())
        max_parents = max(
            len(prs) for gs in self.graph_structs for prs in gs.parents.values())
        self._max_bi_relations = max(max_children, max_parents)
        return self._max_bi_relations

    def get_max_treelets(self):
        if self._max_treelets is not None:
            return self._max_treelets
        self._max_treelets = max(
            len(treelets) for gs in self.graph_structs for treelets in itertools.chain(
            gs.treelets_predicate.values(),
            gs.treelets_right.values(),
            gs.treelets_left.values()))
        return self._max_treelets

    def make_vocabulary(self):
        if self.word2ind is not None and len(self.word2ind) > 1:
            logging.info('word2ind already exists. Reusing it.')
            return self.word2ind
        counter = Counter()
        constants = []
        special = []
        for gs in self.graph_structs:
            graph = gs.graph
            for nid in graph.nodes:
                token = get_node_token(graph, nid)
                if get_label(graph, nid, 'type') == 'constant':
                    constants.append(token)
                    counter[token] += 1
                else:
                    special.append(token)
                    counter[token] += 1
        logging.info('Most common 10 tokens: {0}'.format(counter.most_common()[:10]))
        special = sorted(set(special))
        logging.info('Got {0} special tokens: {1}'.format(len(special), special))
        constants = sorted(set(constants))
        logging.info('Got {0} constant tokens. Some of them are: {1}'.format(
            len(constants), constants[:10]))
        vocab = special + constants
        assert '<unk>' not in vocab
        [self.word2ind[w] for w in vocab]
        return self.word2ind

    # TODO: guard against index-out-of-bounds error when preparing trial and
    # test matrices.
    def make_birel_matrix(self, relation='children'):
        birel = np.zeros((
            len(self.graph_structs),
            self._max_nodes,
            self._max_bi_relations,
            2),
            dtype='int32')
        for i, gs in enumerate(self.graph_structs):
            for j, nid in enumerate(gs.graph.nodes):
                nid_token = get_node_token(gs.graph, nid)
                for k, rel_nid in enumerate(getattr(gs, relation)[nid]):
                    rel_token = get_node_token(gs.graph, rel_nid)
                    try:
                        birel[i, j, k, :] = [nid, rel_nid]
                    except IndexError:
                        continue
        return birel

    # TODO: remove word2ind mapping.
    def make_treelet_matrix(self, relation='treelet_predicate'):
        treelets = np.zeros((
            len(self.graph_structs),
            self._max_nodes,
            self._max_treelets,
            3),
            dtype='int32')
        for i, gs in enumerate(self.graph_structs):
            for j, nid in enumerate(gs.graph.nodes):
                nid_token = get_node_token(gs.graph, nid)
                for k, (rel1_nid, rel2_nid) in enumerate(getattr(gs, relation)[nid]):
                    rel1_token = get_node_token(gs.graph, rel1_nid)
                    rel2_token = get_node_token(gs.graph, rel2_nid)
                    treelets[i, j, k, :] = [
                        self.word2ind[nid_token],
                        self.word2ind[rel1_token],
                        self.word2ind[rel2_token]]
        return treelets

    def make_birel_normalizers(self, relation='children'):
        birel_norm = np.zeros((
            len(self.graph_structs),
            self._max_nodes,
            self._max_bi_relations),
            dtype='float32')
        for i, gs in enumerate(self.graph_structs):
            for j, nid in enumerate(gs.graph.nodes):
                degree = len(gs.children[nid]) + len(gs.parents[nid])
                rel_degree = len(getattr(gs, relation)[nid])
                for k in range(rel_degree):
                    birel_norm[i, j, k] = 1. / degree
        return birel_norm
            
    def make_treelets_normalizers(self):
        treelets_norm = np.ones((
            len(self.graph_structs),
            self._max_nodes,
            1),
            dtype='float32')
        for i, gs in enumerate(self.graph_structs):
            for j, nid in enumerate(gs.graph.nodes):
                num_treelets = sum([
                    len(getattr(gs, 'treelets_' + d)[nid]) for d in ['predicate', 'right', 'left']])
                if num_treelets == 0.0:
                    treelets_norm[i, j, 0] = 0.0
                else:
                    treelets_norm[i, j, 0] = 1. / num_treelets
        return treelets_norm

    def make_node_inds(self):
        node_inds = np.zeros((
            len(self.graph_structs),
            self._max_nodes),
            dtype='float32')
        for i, gs in enumerate(self.graph_structs):
            for j, nid in enumerate(gs.graph.nodes):
                node_token = get_node_token(gs.graph, nid)
                node_inds[i, nid] = self.word2ind[node_token]
        return node_inds

    def make_node_embeddings(self):
        embeddings = np.random.uniform(size=(
            len(self.word2ind), self.emb_dim))
        # embeddings = np.array(range(len(self.word2ind) * self.emb_dim), dtype='float32').reshape(
        #     len(self.word2ind), self.emb_dim)
        # embeddings[self.word2ind['<&>'], :] *= 100
        embeddings[0, :] *= 0.0 # reserved for "no-word" (padding).
        # embeddings[self.word2ind['<unk>'], :] *= 0
        return embeddings

    def make_matrices(self):
        self._max_nodes = self.get_max_nodes()
        self._max_bi_relations = self.get_max_bi_relations()
        self._max_treelets = self.get_max_treelets()
        logging.info('Max nodes: {0}'.format(self._max_nodes))
        logging.info('Max bi-relations: {0}'.format(self._max_bi_relations))
        logging.info('Max treelets: {0}'.format(self._max_treelets))

        # Populates self.word2ind
        self.make_vocabulary()

        if self.node_embs is None:
            self.node_embs = self.make_node_embeddings()
        self.node_inds = self.make_node_inds()
        # Makes relations between pairs of nodes (children and parents).
        self.children = self.make_birel_matrix(relation='children')
        self.parents = self.make_birel_matrix(relation='parents')

        # Makes relations between three nodes (treelets).
        self.treelets_predicate = self.make_treelet_matrix(relation='treelets_predicate')
        self.treelets_right = self.make_treelet_matrix(relation='treelets_right')
        self.treelets_left = self.make_treelet_matrix(relation='treelets_left')

        # Makes normalizers (numbers between 0 and 1) to weight the sum
        # and obtain average embeddings.
        self.birel_child_norm = self.make_birel_normalizers(relation='children')
        self.birel_parent_norm = self.make_birel_normalizers(relation='parents')
        self.treelets_norm = self.make_treelets_normalizers()
        return None

