import numpy as np
import pandas as pd
import sys
sys.path.append('/mnt/projects/sne_students/M.Thesis_Scott/git_repos/GNN_MBTR_MD/graphNN_tools/')
import gnn_tools as gnn
import subprocess
import pickle
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import time
from torch_geometric.data import InMemoryDataset 
######################### INITIALIZE SOME VALUES
end = 2000000
show_plots = 0
num_epochs = 300
GNN = 1
MD = 0
MBTR = 0

import matplotlib.pyplot as plt
########

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
######################### READ DATASET
if (0 == MBTR):
    dataset = transferIntegral(root="../data/")
    dataset = dataset.shuffle()
    dataset = dataset[:end]
else:
    with open("../data/processed_normal/dataset.pic", 'rb') as filehandle:
        dataset = pickle.load(filehandle)
    dataset = dataset.shuffle()
    dataset, temp = torch.utils.data.random_split(dataset,(end,(len(dataset)-end)))

######################### SET UP INITIAL PARAMETERS
#for target_term in ['homo', 'zpve', 'mu', 'alpha','r2', 'U0', 'Cv', 'omega']:
for target_term in ['transfer_integral']:
    print(target_term)       
    # Total energies PBE
    if ("transfer_integral" == target_term): 
        hyper = 0
        #param_best = [0.007063748758319851, 16, 10, 16, 9, 21]
        #            [lr, batch_size, # of neurons, # of neurons , md_param, mbtr_param] 
        param_best = [0.005, 64, 28, 28, 9, 21]
        param_range = [[0.001, 0.01],[64, 64], [16, 64], [16, 64], [8, 32], [8, 45]]
    param_initial = param_best 
    
    ######################### SOME PLOTS
    
    #gnn.plot_target_vs_avg_edge_attr(dataset)
    #gnn.plot_target_vs_features(df_reduced["mw"], np.log10(df_reduced[target_term]))
    
    ######################### HYPER PARAMETER OPTIMIZATION
    if ( 1 == hyper):
        # Clean the hyper data
        subprocess.run(["rm", "-r", "hyper"]) 
        subprocess.run(["mkdir", "hyper"])
        with open("./hyper/dataset.pic", 'wb') as filehandle:
                    pickle.dump(dataset, filehandle, protocol=4)
        '''
        # SIMPLE
        param_best = gnn.fit_hyperParameters_simple(target_term, 0.7, 50, 5, param_range, param_initial, 1, 1, 1, show_plots) 
        # Now clean the hyper data            
        subprocess.run(["rm", "-r", "hyper"])
        if (1 == show_plots):
            print("creating plots for hyper parameter optimization")
            for i in [1,2]:
                gnn.plot_hyper(target_term,i)
        '''        
        # RANDOM SEARCH
        #hyper_batch_size, target_term, dataset1, split_size, parameter_ranges, nbrTrials, nbrEpochs, MD
        param_best, param_best_5 = gnn.fit_hyperParameters_random(1, target_term, 0.75, param_range, 7, 15, GNN, MD, MBTR)         
    
        # We can use machine larning to predict lowest loss with the hyperparameters as features 
        #param_best_ml = gnn.fit_ML_hyper_rand(target_term, param_range)
    with open("../results/all_hyperparameters.txt", "a") as file_object:
        file_object.write("%s = %s   (GNN = %s  MD = %s  MBTR = %s)\n" % (target_term, param_best, GNN, MD, MBTR))
    ######################### FINAL OPTIMIZATION
    #dataset_small = dataset[:nbrData_hyper]

    # GET RESULTS: GNN AND GNN + MOLECULAR DESCRIPTORS

    print("########## ",target_term," GNN =", GNN, "MD = ",MD, "MBTR =", MBTR, "#############")
    print("Molecular Descriptor used =", MD)
    # getloss, verbose, target_term, dataset, split_size, num_epochs, lr, batch_size,  p1, p2, numLayer, numFinalFeature, GNN, MD, MBTI
    trainData, testData = gnn.fit_GNN(0, 1, target_term, dataset, 0.75, num_epochs, *param_best, GNN, MD, MBTR)
    trainData.to_csv("../results/%s/train_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    testData.to_csv("../results/%s/test_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    trainData = pd.read_csv("../results/%s/train_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))
    testData = pd.read_csv("../results/%s/test_CNN=%s_MD=%s_MBTR=%s.csv" % (target_term, GNN, MD, MBTR))        

    gnn.plot_losses(target_term, GNN, MD, MBTR)
    plt.savefig("losses.png")
    plt.clf()
    time.sleep(0.1)
    gnn.plot_results(trainData, testData, target_term, show = 1)
    plt.savefig("results.png")

#   if (1 == show_plots):
#       for i in range(num_epochs):
#           gnn.plot_losses(target_term, GNN, MD, MBTR)
#           time.sleep(30)
#       gnn.plot_results(trainData, testData, target_term, show = show_plots)
    
    # Now store the final result
    MAE = mean_absolute_error(testData["Preds"].to_numpy(), testData["Target"].to_numpy())
    with open("../results/all_results.txt", "a") as file_object:
        file_object.write("%s = %s    (GNN = %s  MD = %s  MBTR = %s)\n" % (target_term, MAE, GNN, MD, MBTR))

