import os
import pathlib
import pickle

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.datasets import TUDataset
import networkx as nx

from gudhi.representations import Landscape

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from src.datasets.TDA import get_barcode


class SpectreGraphDataset(InMemoryDataset):
    """ Graph datasets used in the paper: comm20 (Community-small), ego_small,
        enzymes (ENZYMES), planar and sbm.
        Each processed sample additionally holds the homological feature vector
        mu_{G_0} (Eq. 6, persistence landscape of the 0-dim persistence diagram),
        pre-computed here so that training needs no extra TDA computation. """

    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def download(self):
        """ Collect the raw adjacency matrices and split them into train/val/test. """
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]

        if self.dataset_name in ['sbm', 'planar', 'comm20']:
            if self.dataset_name == 'sbm':
                raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
            elif self.dataset_name == 'planar':
                raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
            else:
                raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
            file_path = download_url(raw_url, self.raw_dir)
            adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        elif self.dataset_name == 'enzymes':
            dataset = TUDataset(root=os.path.join(self.raw_dir, 'TU'), name='ENZYMES')
            adjs = []
            for data in dataset:
                G = torch_geometric.utils.to_networkx(data, to_undirected=True)
                adjs.append(torch.Tensor(nx.adjacency_matrix(G).toarray()))

        elif self.dataset_name == 'ego_small':
            file_path = os.path.join(base_path, 'data', 'ego_small', 'ego_small.pkl')
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f'ego_small raw file not found at {file_path}. '
                    'Place ego_small.pkl (200 small ego subgraphs of Citeseer) at that path.')
            with open(file_path, 'rb') as f:
                graphs = pickle.load(f)
            adjs = [torch.Tensor(nx.adjacency_matrix(g).toarray()) for g in graphs]

        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        num_graphs = len(adjs)
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len
        indices = torch.randperm(num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        LS = Landscape(num_landscapes=4, resolution=16)     # L = 4, S = 16  ->  mu_G0 in R^64 (= model dv)

        for adj in raw_dataset:
            if adj.sum() == 0:      # skip empty graphs
                continue
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)

            # homological feature vector mu_G0: persistence landscape of the finite barcodes
            barcode = get_barcode(adj)
            if barcode is None:
                continue
            finite_barcodes = barcode[torch.isfinite(barcode[:, 1])]
            if len(finite_barcodes) == 0:
                continue
            ls = LS.fit_transform([finite_barcodes.numpy()])

            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes,
                                             vectorization=torch.Tensor(ls))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                               split='val', root=root_path),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                split='test', root=root_path)}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
