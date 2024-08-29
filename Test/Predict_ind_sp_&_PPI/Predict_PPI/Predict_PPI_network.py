import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from gensim.models import word2vec
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################################################################################
from attention import Self_cross_attn
from construct_dataset import construct_dataset
####################################################################################################################


def w2v_dict_values(sequences, enc_model, k_mer):
    sequences = list(set(sequences))
    seq_dict = {}
    for i in range(len(sequences)):
        seq_dict[sequences[i]] = torch.tensor([enc_model.wv[sequences[i][j: j + k_mer]] for j in range(len(sequences[i]) - k_mer + 1)])

    return seq_dict

class Attnseq_PPI_model():
    def __init__(self,deep_path, enc_dict):
        self.enc_model = enc_dict
        self.thresh = 0.5
        self.deep_path = deep_path
        
    def model_testing_PPI(self, val_data_sets):
        val_data = construct_dataset(val_data_sets, seq_encoding = self.enc_model, seq_max_len = 8000, window = 20, stride = 10, k_mer = 4)
        val_loader = DataLoader(dataset = val_data,shuffle=False)
        self.model = Self_cross_attn().to(device)
        self.model.load_state_dict(torch.load(self.deep_path, map_location = device))
        self.criterion = torch.nn.BCELoss()
        
        validation_losses, val_probs, val_labels = [], [], []
        self.model.eval()
        for i, (protein_1, protein_2, protein_msk_1, protein_msk_2, labels) in enumerate(val_loader):
            with torch.no_grad():
                probs = self.model(protein_1, protein_2, protein_msk_1, protein_msk_2)
                loss = self.criterion(probs, labels)
                validation_losses.append(loss)
                val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())

        thresh = 0.5
        val_probs = (np.array(val_probs) + 1 -thresh).astype(np.int16)
        val_probs = pd.DataFrame(val_probs, columns=['predictions'])
        total_ones = val_probs.values.sum()
        return val_probs, total_ones


def test_ind_sp_ondata(dataset, test_data_name):
    PN = pd.read_csv( dataset)
    ################################
    PN0 = PN['col1']
    PN1 = PN['col2']
    interaction = PN['interaction']
    val_data1 = pd.concat((PN0,PN1,interaction),axis=1)

    ### ind_sp + PPI data word2vec #######################################
    w2v_path = r"Datasets_&_w2v_models\Individual_species_and_PPI_data\ISP_PPI_w2v\word2vec_model.pt"
    ######################################################################
    ## Import model 
    model_path = r"Test\Predict_ind_sp_&_PPI\Trained_model\model"

    def Testing_model_PPI_Pred(val_data, w2v_path,deep_path, data_name):
        print(f"Predicting {data_name} dataset....", flush = True)
        
        validation_data = val_data
        w2v_model = word2vec.Word2Vec.load(w2v_path)
        mat_dict = w2v_dict_values(validation_data["col1"].values.tolist() + validation_data["col2"].values.tolist(), enc_model=w2v_model, k_mer=4)
        net = Attnseq_PPI_model(enc_dict=mat_dict,deep_path=deep_path)
        total_data,correct_pred = net.model_testing_PPI(validation_data)
        
        print(f'correct prediction {correct_pred}/{len(total_data)}')
        accuracy = 100 * correct_pred / len(total_data)
        print(f'Accuracy:{accuracy} %')

    Testing_model_PPI_Pred(val_data=val_data1, w2v_path=w2v_path, deep_path=model_path,data_name=test_data_name)


csv_files = {
    "CD9": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\PPI_network\cd9\CD9.csv",
    "crossover_network": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\PPI_network\crossover\Crossovernetwork.csv",
}


for name, file in csv_files.items():
    test_ind_sp_ondata(file, name)
    
    
    
    
    
