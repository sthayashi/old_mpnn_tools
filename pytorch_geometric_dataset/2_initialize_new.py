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

from mendeleev import element
from sklearn.preprocessing import StandardScaler
import subprocess

#with open('mol_name.txt', 'r') as f:
#    mol_name = f.read()

with h5py.File("parsed_molecules.h5", "r") as input:
    print(input.keys())
    mol_name = list(input.keys())[0]
    for j in [mol_name]:
        training = input.get(j)
        print(training.keys())
        coulomb_interactions = training.get("coulomb_interaction")[()]
        TI = training.get("transfer_integrals")[()]

        atom_types = training.get("atom_types")[()]
        atomic_numbers = training.get("atomic_numbers")[:]

        # number of atoms in organic molecule
        nbrAtoms = len(atom_types)

        x = training.get("x")[:]

coulomb_interactions = coulomb_interactions.reshape(coulomb_interactions.shape[0], coulomb_interactions.shape[1]*coulomb_interactions.shape[2])
TI = TI.reshape(TI.shape[0],1)
#################

TotalDataPoints = coulomb_interactions.shape[0]

################

x = torch.tensor(x, dtype=torch.float)

#Construct a dataset for dimers with edge_attributes and targets (TI)

ind1 = []

for i in range(nbrAtoms):
    for j in range(2):
        arr = i # for instance
        ind1.append(arr)
    
ind1 = np.array(ind1)

#print("ind1 =",ind1)

ind2 = []

for i in range(nbrAtoms, 2 * nbrAtoms):
    for j in range(nbrAtoms,nbrAtoms + 2):
        arr = j # for instance
        ind2.append(arr)
    
ind2 = np.array(ind2)

#print("ind2 =",ind2)

edge_index = torch.tensor([ind1,ind2], dtype=torch.long)

#for i in range(dimer.shape[0]):
#    print(i)

#########################


features = torch.FloatTensor(coulomb_interactions)

features = StandardScaler().fit_transform(features)
features = torch.tensor(features)
features = features.to(torch.float32)

all_ti = torch.FloatTensor(TI)

with open('features.pickle', 'wb') as handle:
    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    
with open('all_ti.pickle', 'wb') as handle:
    pickle.dump(all_ti, handle, protocol=pickle.HIGHEST_PROTOCOL) 


################
# Now define a function to create a pytorch geometric dataset

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
        
        data_list = []

        for i in range(TotalDataPoints): 
            #print("i =",i)
            edge_attr = [] 
            for j in range(1):
                edge_attr.append([])    
            edge_attr[0].append(features[i])
            #edge_attr = np.array(edge_attr).T
            edge_attr = edge_attr[0][0].T
            sizeT = edge_attr.size()[0]
            edge_attr = torch.reshape(edge_attr, (sizeT,1))
            y = all_ti[i]
            y = abs(y)
            #print("y=",y)
            # Remove the cases where TI is too high (QM calculation did not converge)
            if y < 10000:
                y = torch.log10(y)
                    #y = y.view(1,1)
                    #print(y.shape)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                #print(data.y)
                data_list.append(data)

        
        #print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

subprocess.run(["rm", "-r", "../data/processed"])        
dataset = transterIntegral(root='../data/')   

#############

