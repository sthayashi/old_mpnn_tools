import torch
import numpy as np
import pandas as pd
import itertools
import math
import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import NNConv, GATConv, Set2Set
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset  
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU, GRU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops,add_remaining_self_loops
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from inits import reset, uniform
    
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d as BN
from torch.nn import LayerNorm as LN
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch_geometric.data import DataLoader
from pathlib import Path


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        nn (torch.nn.Module) – A neural network ℎ𝚯 that maps edge features 
        edge_attr of shape [-1, num_edge_features] to 
        shape [-1, in_channels * out_channels], e.g., 
        defined by torch.nn.Sequential.
        '''   
        # number of neurons in dense nn
        p1 = 28
        p2 = 28
        # get features from dataset, but hardcode this for now
        nbr_node_features = dataset[0].x.size()[1]
        nbr_edge_features = dataset[0].edge_attr.size()[1]

        totNbrFeatures = 0  
        totNbrFeatures += p2*2

        self.lin0 = torch.nn.Linear(nbr_node_features, p2, bias = False)
        self.BN0 = BN(round(p2))

        nn = Seq(Linear(nbr_edge_features, p1, bias = False), BN(p1), LeakyReLU(), Linear(p1, p2 * p2, bias = False), BN(p2 * p2))

        self.conv = NNConv(p2, p2,nn, aggr='mean')

        self.set2set = Set2Set(p2, processing_steps=3)    
        self.gru = GRU(p2, p2)       

        print("totNbrFeatures", totNbrFeatures) 
        self.lin1 = torch.nn.Linear(totNbrFeatures, round(totNbrFeatures/2))  
        self.lin2 = torch.nn.Linear(round(totNbrFeatures/2), round(totNbrFeatures/4))
        self.lin_final = torch.nn.Linear(round(totNbrFeatures/4), 1)
        
    def forward(self, data):
        # Linear -> BN -> relu
        #                   |--> nnconv -> relu -> GRU
        #                                           |--> set2set -> Linear -> Linear -> Linear
        y = None
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        out = F.leaky_relu(self.BN0(self.lin0(x)))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.leaky_relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        y_gnn = self.set2set(out, batch)  
        y = y_gnn

        y = F.dropout(y, p = 0.5, training=self.training)
        y = F.leaky_relu(self.lin1(y)) 
        y = F.leaky_relu(self.lin2(y))
        y = self.lin_final(y)
        y = y.squeeze(-1)
        return y   

# Training loop

def train(epoch, train_loader, train_dataset):
    model.train()
    loss_all = 0
    for data in train_loader:
        #print("in for loop of train_loader")
        data = data.to(device)
        #print("train definition, edge_attr =",data.edge_attr.size())
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        if (1 == sel_target):
            label = label.reshape(-1,numTarget)
            label = label[:,targetNbr]
        if (1 == scale_target):
            label = (label - y_mean) / y_std
        loss = criterion(output, label)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss_all += data.num_graphs * loss.item()                    
        optimizer.step()
        del loss, output, label
    return loss_all / len(train_dataset)

def test(loader):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            label = data.y.to(device)
            if (1 == sel_target):
                label = label.reshape(-1,numTarget)
                label = label[:,targetNbr]
            if (1 == scale_target):
                label = (label - y_mean) / y_std
            error += (model(data) - label).abs().sum().item()  # MAE
    return error / len(loader.dataset)

def evaluate(loader,whichDataset):   
    model.eval()
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            label = data.y.to(device)
            label = label.reshape(-1,numTarget)
            label = label[:,targetNbr]
            label = (label - y_mean) / y_std
            loss = criterion(output, label)
            loss_all += data.num_graphs * loss.item()
    return loss_all / len(whichDataset)

def get_results(train_loader, test_loader): 
    # for training
    # what is going on with this y_std nonsense?
    preds = []
    target = []     
    for data in train_loader:
        data = data.to(device)
        output = model(data).detach()
        output = output.cpu()  
        label = data.y.to(device)
        label = label.reshape(-1,numTarget)
        label = label[:,targetNbr]
        label = label.cpu()
        preds.append(output * y_std + y_mean)
        target.append(label) 
    preds = torch.cat(preds,0)
    target = torch.cat(target,0)
    trainData = pd.DataFrame({'Target': target, 'Preds': preds})
    
    # for test 
    preds = []
    target = []     
    for data in test_loader:
        data = data.to(device)
        output = model(data).detach()
        output = output.cpu()  
        label = data.y.to(device)
        label = label.reshape(-1,numTarget)
        label = label[:,targetNbr]
        label = label.cpu()
        preds.append(output * y_std + y_mean)
        target.append(label)   
    preds = torch.cat(preds,0)
    target = torch.cat(target,0)
    testData = pd.DataFrame({'Target': target, 'Preds': preds})

    return(trainData, testData)

def evaluate(model, data_loader, dataset):
    model.eval()
    criterion = torch.nn.MSELoss()
    loss_all = 0
    with torch.no_grad():
        optimizer.zero_grad()
        for data in data_loader:
            data = data.to(device)
            y_hat = model(data)
            y = data.y.to(device)
            loss = criterion(y_hat, y)
            loss_all += data.num_graphs * loss.item()
    return loss_all / len(dataset)


if __name__ == '__main__':
    exec(open("dataset_class.py").read())
    dataset = transferIntegral(root="data")

    CHECKPOINT_PATH = Path("checkpoint/checkpoint")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if CHECKPOINT_PATH.is_file():
        print("Found model...")
        model = Net()
        model.load_state_dict(torch.load(str(CHECKPOINT_PATH), map_location=device))
    else:
        print("Starting model from scratch on {} ...".format(device))
        model = Net().to(device)

    learning_rate = 0.005

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=True)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)
