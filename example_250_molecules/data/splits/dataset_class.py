import sys
sys.path.append('/mnt/projects/sne_students/M.Thesis_Scott/git_repos/GNN_MBTR_MD/graphNN_tools/')
import gnn_tools as gnn

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset

# Dataset class
class transferIntegral(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(transferIntegral, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['processed_data.dataset']
    def download(self):
        pass
    def process(self):
        pass

#dataset = transferIntegral(root='../data')
#dataset = dataset.shuffle()
#dataset = dataset[:end]
