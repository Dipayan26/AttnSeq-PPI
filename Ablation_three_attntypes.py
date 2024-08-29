
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from gensim.models import word2vec
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################################################################################
from construct_dataset import construct_dataset
####################################################################################################################
from attention_ablation import only_self_attn,only_cross_attn, NO_attention

class only_self_attn_code():
    def __init__(self,deep_path, enc_dict):
        self.enc_model = enc_dict
        self.thresh = 0.5
        self.deep_path = deep_path
        
    def model_testing_PPI(self, val_data_sets):
        val_data = construct_dataset(val_data_sets, seq_encoding = self.enc_model, seq_max_len = 8000, window = 20, stride = 10, k_mer = 4)
        val_loader = DataLoader(dataset = val_data,shuffle=False)
        self.model = only_self_attn().to(device)
        self.model.load_state_dict(torch.load(self.deep_path, map_location = device))
        self.criterion = torch.nn.BCELoss()
        validation_losses, val_probs, val_labels = [], [], []
        self.model.eval()
        for i, (protein_1, protein_2, attn_msk_1, attn_msk_2, labels) in enumerate(val_loader):
            with torch.no_grad():
                probs = self.model(protein_1, protein_2, attn_msk_1, attn_msk_2)
                
                loss = self.criterion(probs, labels)

                validation_losses.append(loss)
                val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())

        thresh = 0.5
        val_probs = (np.array(val_probs) + 1 -thresh).astype(np.int16)
        val_probs = pd.DataFrame(val_probs, columns=['predictions'])
        total_ones = val_probs.values.sum()
        return val_probs, total_ones

class only_cross_attn_code():
    def __init__(self,deep_path, enc_dict):
        self.enc_model = enc_dict
        self.thresh = 0.5
        self.deep_path = deep_path
        
    def model_testing_PPI(self, val_data_sets):
        val_data = construct_dataset(val_data_sets, seq_encoding = self.enc_model, seq_max_len = 8000, window = 20, stride = 10, k_mer = 4)
        val_loader = DataLoader(dataset = val_data,shuffle=False)
        self.model = only_cross_attn().to(device)
        self.model.load_state_dict(torch.load(self.deep_path, map_location = device))
        self.criterion = torch.nn.BCELoss()
        
        validation_losses, val_probs, val_labels = [], [], []
        self.model.eval()
        for i, (protein_1, protein_2, attn_msk_1, attn_msk_2, labels) in enumerate(val_loader):
            with torch.no_grad():
                probs = self.model(protein_1, protein_2, attn_msk_1, attn_msk_2)
                
                loss = self.criterion(probs, labels)

                validation_losses.append(loss)
                val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())

        thresh = 0.5
        val_probs = (np.array(val_probs) + 1 -thresh).astype(np.int16)
        val_probs = pd.DataFrame(val_probs, columns=['predictions'])
        total_ones = val_probs.values.sum()
        return val_probs, total_ones

class NO_attention_code():
    def __init__(self,deep_path, enc_dict):
        self.enc_model = enc_dict
        self.thresh = 0.5
        self.deep_path = deep_path
        
    def model_testing_PPI(self, val_data_sets):
        val_data = construct_dataset(val_data_sets, seq_encoding = self.enc_model, seq_max_len = 8000, window = 20, stride = 10, k_mer = 4)
        val_loader = DataLoader(dataset = val_data,shuffle=False)
        self.model = NO_attention().to(device)
        self.model.load_state_dict(torch.load(self.deep_path, map_location = device))
        self.criterion = torch.nn.BCELoss()
        
        validation_losses, val_probs, val_labels = [], [], []
        self.model.eval()
        for i, (protein_1, protein_2, attn_msk_1, attn_msk_2, labels) in enumerate(val_loader):
            with torch.no_grad():
                probs = self.model(protein_1, protein_2, attn_msk_1, attn_msk_2)
                
                loss = self.criterion(probs, labels)

                validation_losses.append(loss)
                val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())

        thresh = 0.5
        val_probs = (np.array(val_probs) + 1 -thresh).astype(np.int16)
        val_probs = pd.DataFrame(val_probs, columns=['predictions'])
        total_ones = val_probs.values.sum()
        return val_probs, total_ones

