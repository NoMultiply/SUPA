import numpy as np
import os


class SUPADataset:
    def __init__(self, root):
        self.root = root
        self.train_edges = self.load_edges(os.path.join(root, 'train.txt'))
        self.valid_edges = self.load_edges(os.path.join(root, 'valid.txt'))
        self.test_edges = self.load_edges(os.path.join(root, 'test.txt'))
        self.node_types = {}
        with open(os.path.join(root, 'node_types.txt')) as fin:
            for line in fin:
                nid, tid = map(int, line.strip().split())
                self.node_types[nid] = tid

        start_time = self.train_edges[0][-1]
        self.train_edges[:, -1] = self.train_edges[:, -1] - start_time
        self.valid_edges[:, -1] = self.valid_edges[:, -1] - start_time
        self.test_edges[:, -1] = self.test_edges[:, -1] - start_time

        self.n_nodes = len(self.node_types)
        self.n_node_types = len(set(self.node_types.values()))
        self.edge_types = set(self.train_edges[:, 2])
        self.n_edge_types = len(self.edge_types)

    @staticmethod
    def load_edges(filename):
        edges = []
        with open(filename) as fin:
            for line in fin:
                edges.append(list(map(int, line.strip().split())))
        return np.array(edges)

    def print_info(self):
        print('============================================')
        print('Dataset:', self.root)
        print('# of node types:', self.n_node_types)
        print('# of edge types:', self.n_edge_types)
        print('# of nodes:', self.n_nodes)
        print('# of train edges:', self.train_edges.shape[0])
        print('# of valid edges:', self.valid_edges.shape[0])
        print('# of test edges:', self.test_edges.shape[0])
        print('Max time delta:', self.test_edges[-1][-1])
        print('============================================')
