import torch
import numpy as np
import pandas as pd
import itertools
import math
import pickle
import h5py
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool)

from sklearn.preprocessing import StandardScaler
import subprocess

def make_data_list():
    with h5py.File("parsed_molecules.h5", "r") as input:
        print(input.keys())
        mol_names = list(input.keys())
    
        num_of_organic_molecules = len(mol_names)
    
        data_list = []
    
        for mol in mol_names:
            training = input.get(mol)
    
            o2_good_inds = training.get("o2_good_inds")[:]
    
            # node features already includes O2
            x = torch.tensor(training.get("x")[:], dtype=torch.float)
    
            # Coulomb interactions ready to be directly plugged into Data
            edge_indices_i = training.get("edge_indices_i")[:]
            edge_indices_j = training.get("edge_indices_j")[:]
            edge_index = torch.tensor([edge_indices_i, edge_indices_j], dtype=torch.long)
    
            coulomb_interactions = training.get("coulomb_interactions")[:]
            
            # Transfer integral
            TI = training.get("transfer_integrals")[o2_good_inds]
            assert TI.shape[0] == coulomb_interactions.shape[0], "Unexpected number of transfer integrals / coulomb interactions"
            for i in range(TI.shape[0]):
                edge_attr = coulomb_interactions[i, :].reshape(-1, 1)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
                edge_attr = StandardScaler().fit_transform(edge_attr)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
                y = torch.tensor(TI[i]).reshape(-1)
                # NOTE SH: This abs is suspicious
                y = abs(y)
                y = torch.log10(y)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                data_list.append(data)

    return data_list

class transterIntegral(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(transterIntegral, self).__init__(root, transform, pre_transform)
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
        data_list = make_data_list()
       #data_list = [data for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

subprocess.run(["rm", "-r", "../data/processed"])        
dataset = transterIntegral(root='../data/')   
