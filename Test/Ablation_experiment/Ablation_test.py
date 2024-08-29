
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
import sklearn.metrics as metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################################################################################
from Ablation_three_attntypes import NO_attention_code, only_cross_attn_code, only_self_attn_code

def w2v_dict_values(sequences, enc_model, k_mer):
    sequences = list(set(sequences))
    seq_dict = {}
    for i in range(len(sequences)):
        seq_dict[sequences[i]] = torch.tensor([enc_model.wv[sequences[i][j: j + k_mer]] for j in range(len(sequences[i]) - k_mer + 1)])

    return seq_dict

def output_csv(filename, data):
    data.to_csv(filename, index=False)

def test_ind_sp_ondata(dataset, test_data_name, mod1_path, code):
    PN = pd.read_csv( dataset)
    ################################
    PN0 = PN['col1']
    PN1 = PN['col2']
    interaction = PN['interaction']
    val_data1 = pd.concat((PN0,PN1,interaction),axis=1)

    ### ind_sp + PPI data word2vec #######################################
    w2v_path = r"Datasets_&_w2v_models\Individual_species_and_PPI_data\ISP_PPI_w2v\word2vec_model.pt"
    ######################################################################
    model_path_1 = mod1_path
    def Testing_model_PPI_Pred(val_data, w2v_path,deep_path, data_name):
        print(f"Predicting {data_name} dataset....", flush = True)
        
        validation_data = val_data
        w2v_model = word2vec.Word2Vec.load(w2v_path)
        mat_dict = w2v_dict_values(validation_data["col1"].values.tolist() + validation_data["col2"].values.tolist(), enc_model=w2v_model, k_mer=4)
        net = code(enc_dict=mat_dict,deep_path=deep_path)
        out,tot_one = net.model_testing_PPI(validation_data)
        
        print(f'correct prediction {tot_one}/{len(out)}')
        accuracy = 100 * tot_one / len(out)
        print(f'Accuracy:{accuracy} %')
        print('-----------------------')

    Testing_model_PPI_Pred(val_data=val_data1, w2v_path=w2v_path, deep_path=model_path_1,data_name=test_data_name)



Data_path = {
    "M_musculus": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\M_musc\M_musc.csv",
    "Homo_sapiens": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\H_sapiens\H_sapiens.csv",
    "E_coli": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\E_coli\E_coli.csv",
    "c_elegans": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\C_legans\c_legans.csv",
    "CD9": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\PPI_network\cd9\CD9.csv",
    "crossover_network": r"Datasets_&_w2v_models\Individual_species_and_PPI_data\PPI_network\crossover\Crossovernetwork.csv",
}


Model_path = {
    "NO_attention": r"Test\Ablation_experiment\Trained_models\No_attention\model",
    "only_cross_attention": r"Test\Ablation_experiment\Trained_models\Only_cross\model",
    "only_Self_attention": r"Test\Ablation_experiment\Trained_models\Only_self\model",
}

for model_name, mod_path in Model_path.items():
    print("===================================================")
    print(f'Testing with Model __{model_name}__')
    print("===================================================")
    
    if model_name =='only_Self_attention':

        for name, file in Data_path.items():
            test_ind_sp_ondata(dataset=file, test_data_name=name, mod1_path=mod_path,  code=only_self_attn_code)
            
    elif model_name =='only_cross_attention':

        for name, file in Data_path.items():
            test_ind_sp_ondata(dataset=file, test_data_name=name, mod1_path=mod_path, code=only_cross_attn_code)
            
    elif model_name =='NO_attention':

        for name, file in Data_path.items():
            test_ind_sp_ondata(dataset=file, test_data_name=name, mod1_path=mod_path, code=NO_attention_code)
    
    else:
        pass        

