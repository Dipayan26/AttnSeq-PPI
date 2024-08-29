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
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################################################################################
from Metrics import metrics_dict
from Metrics import conf_matrix
from attention import Self_cross_attn
from construct_dataset import construct_dataset
####################################################################################################################
accuracy1,sensitivity1,specificity1,mcc1,precision1,NPV1,recall1,f11,auc1,AUPRC1 = [],[],[],[],[],[],[],[],[],[]
accuracy2,sensitivity2,specificity2,mcc2,precision2,NPV2,recall2,f12,auc2,AUPRC2 = [],[],[],[],[],[],[],[],[],[]
####################################################################################################################

class Attnseq_PPI_model():
    def __init__(self,enc_dict,max_epoch=1000, early_stop = 2):
        self.enc_model = enc_dict
        self.max_epoch = max_epoch
        self.thresh = 0.5
        self.early_stop = early_stop
        
    def model_training(self, train_data_sets, val_data_sets):       
        train_data = construct_dataset(train_data_sets, seq_encoding = self.enc_model, seq_max_len = 1500, window = 20, stride = 10, k_mer = 4)
        train_loader = DataLoader(dataset = train_data, batch_size = 26, shuffle=True)
        val_data = construct_dataset(val_data_sets, seq_encoding = self.enc_model, seq_max_len = 1500, window = 20, stride = 10, k_mer = 4)
        val_loader = DataLoader(dataset = val_data, batch_size = 26, shuffle=True)
        self.model = Self_cross_attn().to(device)
        self.opt = optim.Adam(params = self.model.parameters(), lr = 0.0001)
        self.criterion = torch.nn.BCELoss()
        max_met = 100
        early_stop_count = 0
        for epoch in range(self.max_epoch):
            training_losses, validation_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []
            self.model.train()
            for i, (protein_1, protein_2, attention_mask_1, attention_mask_2, labels) in enumerate(train_loader):
                self.opt.zero_grad()
                probs = self.model(protein_1, protein_2, attention_mask_1, attention_mask_2)
                loss = self.criterion(probs, labels)
                loss.backward()
                self.opt.step()
                training_losses.append(loss) 
                train_probs.extend(probs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist()) 
                train_labels.extend(labels.cpu().clone().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
            loss_epoch = self.criterion(torch.tensor(train_probs).float(), torch.tensor(train_labels).float())
            print("===========================================", flush = True)
            print("===========================================", flush = True)
            print("training loss:: " + str(loss_epoch), flush = True)
            
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](train_labels, train_probs, thresh = self.thresh)
                else:
                    metrics = metrics_dict[key](train_labels, train_probs)
                print("train_" + key + ": " + str(metrics), flush=True)
                
            tn_tr, fp_tr, fn_tr, tp_tr = conf_matrix(train_labels, train_probs, thresh = self.thresh)
            print("train_true_negative:: value: %f, epoch: %d" % (tn_tr, epoch + 1), flush=True)
            print("train_false_positive:: value: %f, epoch: %d" % (fp_tr, epoch + 1), flush=True)
            print("train_false_negative:: value: %f, epoch: %d" % (fn_tr, epoch + 1), flush=True)
            print("train_true_positive:: value: %f, epoch: %d" % (tp_tr, epoch + 1), flush=True)

            print("------------------------------------------", flush = True)
            print("------------------------------------------", flush = True)
            
            self.model.eval()
            for i, (protein_1, protein_2, attention_mask_1, attention_mask_2, labels) in enumerate(val_loader):
                with torch.no_grad():
                    probs = self.model(protein_1, protein_2, attention_mask_1, attention_mask_2)
                    
                    loss = self.criterion(probs, labels)

                    validation_losses.append(loss)
                    val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
       
            loss_epoch = self.criterion(torch.tensor(val_probs).float(), torch.tensor(val_labels).float())
            
            print("validation loss:: "+ str(loss_epoch), flush = True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](val_labels, val_probs, thresh = self.thresh)
                else:
                    metrics = metrics_dict[key](val_labels, val_probs)
                print("validation_" + key + ": " + str(metrics), flush=True)


            tn_ts, fp_ts, fn_ts, tp_ts = conf_matrix(val_labels, val_probs, thresh = self.thresh)
            
            print("validation_true_negative:: value: %f, epoch: %d" % (tn_ts, epoch + 1), flush=True)
            print("validation_false_positive:: value: %f, epoch: %d" % (fp_ts, epoch + 1), flush=True)
            print("validation_false_negative:: value: %f, epoch: %d" % (fn_ts, epoch + 1), flush=True)
            print("validation_true_positive:: value: %f, epoch: %d" % (tp_ts, epoch + 1), flush=True)
            
            if loss_epoch < max_met:
                early_stop_count = 0
                max_met = loss_epoch
                final_val_probs = val_probs
                final_val_labels = val_labels
                final_train_probs = train_probs
                final_train_labels = train_labels

            else:
                early_stop_count += 1
                if early_stop_count >= self.early_stop:
                    print('Traning parameters not improved from epoch {}\n'.format(epoch + 1 - self.early_stop), flush=True)
                    break

        print(f'Threshold value is {self.thresh}', flush=True)
        
        train_list1 = []
        val_list1 = []
        
        for key in metrics_dict.keys():
            if(key != "auc" and key != "AUPRC" ):
                train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = self.thresh)
                val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = self.thresh)
            else:
                train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                
            print("train_" + key + ": " + str(train_metrics), flush=True)
            print("test_" + key + ": " + str(val_metrics), flush=True)
            
            train_list1.append(train_metrics)
            val_list1.append(val_metrics)
            
        #######################################
        list1 = [accuracy1, sensitivity1, specificity1, precision1, NPV1, recall1, mcc1, f11, auc1, AUPRC1]
        for lst, value in zip(list1, train_list1):
            lst.append(value)
        #######################################
        list2 = [accuracy2, sensitivity2, specificity2, precision2, NPV2, recall2, mcc2, f12, auc2, AUPRC2]
        for lst, value in zip(list2, val_list1):
            lst.append(value)
        #######################################
        return ""

def w2v_dict_values(sequences, enc_model, k_mer):
    sequences = list(set(sequences))
    seq_dict = {}
    for i in range(len(sequences)):
        seq_dict[sequences[i]] = torch.tensor([enc_model.wv[sequences[i][j: j + k_mer]] for j in range(len(sequences[i]) - k_mer + 1)])

    return seq_dict


#########################  HUMAN DATASET ###################################
PN = pd.read_csv( r"Datasets_&_w2v_models\Human\data_1.5k\Human_1.5k.csv")

PN0 = PN['col1']
PN1 = PN['col2']
interaction = PN['interaction']
####################  IMPORT W2V_MODEL   ###########################################################################
w2v_path = r"Datasets_&_w2v_models\Human\w2v_model\AA_model.pt"
####################################################################################################################

kf = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
k_fold_number = 0

for  train1, test1 in kf.split(PN0, interaction):
    
    a1 = PN0[train1]
    a2 = PN1[train1]
    a3 = interaction[train1]
    
    a4= pd.concat((a1,a2,a3),axis=1)

    b1=  PN0[test1]
    b2 = PN1[test1]
    b3 = interaction[test1]
    b4= pd.concat((b1,b2,b3),axis=1)
    
    def k_fold_crossval_model(train_data, val_data, w2v_path):
        print("5 fold cross validation on human dataset............", flush = True)
        training_data = train_data
        validation_data = val_data
        w2v_model = word2vec.Word2Vec.load(w2v_path)
        mat_dict = w2v_dict_values(sequences=training_data["col1"].values.tolist() + validation_data["col1"].values.tolist() + training_data["col2"].values.tolist() + validation_data["col2"].values.tolist(), enc_model=w2v_model, k_mer=4)
        net = Attnseq_PPI_model(mat_dict)
        out = net.model_training(training_data, validation_data)


    k_fold_crossval_model(train_data=a4, val_data=b4, w2v_path=w2v_path)
    
    print('===================================================')
    k_fold_number +=1
    print(f'Done k fold_{k_fold_number}')
    print('===================================================')
    print('===================================================')

####################################################################################################################

metrics_dict1 = {
    'accuracy': accuracy1,'sensitivity': sensitivity1,'specificity': specificity1,'mcc': mcc1,'precision': precision1,'recall': recall1,'NPV': NPV1,'f1': f11,'auc': auc1,'AUPRC': AUPRC1
}

metrics_dict2 = {
    'accuracy': accuracy2,'sensitivity': sensitivity2,'specificity': specificity2,'mcc': mcc2,'precision': precision2,'recall': recall2,'NPV': NPV2,'f1': f12,'auc': auc2,'AUPRC': AUPRC2
}

df = pd.DataFrame(metrics_dict2)
df1 = pd.DataFrame(metrics_dict1)

mean_values = df.mean()
mean_values1 = df1.mean()

df1.loc['mean'] = mean_values1
df.loc['mean'] = mean_values

print('training\n')
print(df1)

print('validation\n')
print(df)

# # Save the DataFrame to a CSV file
# df.to_csv(r'\Five_fold_cross_val.csv')
# df1.to_csv(r'\Five_fold_cross_tr.csv')











