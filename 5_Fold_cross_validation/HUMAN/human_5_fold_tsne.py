#########################################################################

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
# from gensim.models import word2vec
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import  auc

prot_T5_save_path = r"../mat_dict_human_T5.pth"

mat_dict = torch.load(prot_T5_save_path)

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
'''
tensorboard --logdir=runs
'''

####################################################################################################################
accuracy1,sensitivity1,specificity1,mcc1,precision1,NPV1,recall1,f11,auc1,AUPRC1 = [],[],[],[],[],[],[],[],[],[]
accuracy2,sensitivity2,specificity2,mcc2,precision2,NPV2,recall2,f12,auc2,AUPRC2 = [],[],[],[],[],[],[],[],[],[]
#######################################################################################################################################

## Evaliation merics ###########################
def accuracy(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_prob)
def sensitivity(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)
def specificity(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)
def precision(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.precision_score(y_true,y_prob)
def negative_predictive_value(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    npv = tn / (tn + fn)
    return npv
def recall(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.recall_score(y_true,y_prob)
def mcc(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_prob)
def f1(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.f1_score(y_true,y_prob)
def AUPRC(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)
def auc(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.roc_auc_score(y_true, y_prob) 
def conf_matrix(y_true,y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()

    return tn, fp, fn, tp

metrics_dict = {"accuracy":accuracy,"sensitivity":sensitivity, "specificity":specificity ,"precision":precision,'NPV':negative_predictive_value,"recall":recall,"mcc":mcc,"f1":f1,"auc":auc,"AUPRC":AUPRC}


###################################################################################################################
class MultiHeadselfAttention(nn.Module):
    def __init__(self, in_dim, heads, dim_of_d):
        super(MultiHeadselfAttention, self).__init__()
        self.dim  = dim_of_d
        self.d_k = dim_of_d
        self.heads = heads
        self.in_dim = in_dim
        self.Query_lin = nn.Linear(in_dim, self.dim * self.heads, bias=False)
        self.Key_lin = nn.Linear(in_dim, self.dim * self.heads, bias=False)
        self.Value_lin = nn.Linear(in_dim, self.dim * self.heads, bias=False)
        self.out_dense = nn.Linear(self.heads * self.dim, self.in_dim, bias=False)
    def forward(self, Q, K, V, attention_mask):
        Q_skip_conn, batch_size = Q, Q.size(0)
        q_s = self.Query_lin(Q).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        k_s = self.Key_lin(K).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        self.v_s = self.Value_lin(V).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attn, self.v_s)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim)
        output = self.out_dense(output)
        return output + Q_skip_conn

class MultiHeadcrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, heads, dim_of_d):
        super(MultiHeadcrossAttention, self).__init__()
        self.dim  = dim_of_d
        self.d_k = dim_of_d
        self.heads = heads
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.Query_lin = nn.Linear(in_dim1, self.dim * self.heads, bias=False)
        self.Key_lin = nn.Linear(in_dim2, self.dim * self.heads, bias=False)
        self.Value_lin = nn.Linear(in_dim2, self.dim * self.heads, bias=False)
        self.out_dense = nn.Linear(self.heads * self.dim, self.in_dim1, bias=False)
    def forward(self, Q, K, V, attention_mask):
        Q_skip_conn, batch_size = Q, Q.size(0)
        q_s = self.Query_lin(Q).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        k_s = self.Key_lin(K).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        self.v_s = self.Value_lin(V).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(attn, self.v_s)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim)
        output = self.out_dense(output)
        return output + Q_skip_conn

class conv_layers(nn.Module):
    def __init__(self, conv_in, conv_out, kernel_sz, stride, pooling = True, dropout = 0.5):
        super(conv_layers, self).__init__()
        self.pooling = pooling
        self.cnn = nn.Conv1d(conv_in, conv_out, kernel_size = kernel_sz, stride = stride)
        self.pool = torch.nn.MaxPool1d(3, stride = 1, padding=1)
        self.relu_func = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, mat):
        # mat = torch.transpose(mat, -1, -2)
        # mat = mat.permute(0,2,1)
        mat= torch.permute(mat, (0,2,1))
        mat = self.cnn(mat)
        mat = self.relu_func(mat)
        mat = self.dropout_layer(mat)
        mat = self.pool(mat)
        # mat = torch.transpose(mat, -1, -2)
        # mat = mat.permute(0,2,1)
        mat= torch.permute(mat, (0,2,1))
        return mat
#####################################################################################################################################
class HybridPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        max_pooled, _ = torch.max(x, dim=1)
        avg_pooled = torch.mean(x, dim=1)
        hybrid_output = torch.cat([max_pooled, avg_pooled], dim=1)
        return hybrid_output

##################################################################################################
class Self_cross_attn(nn.Module):
    def __init__(self, Input_attn_sz = 100, kernel_sz = 20, stride = 10, heads = 4, d_dim = 32, conv_in_shape = 1024, drop_in_pool = 0.5, drop_in_linear = 0.3):
        super(Self_cross_attn, self).__init__()
        self.conv_1 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.conv_2 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.att_self1 = MultiHeadselfAttention(Input_attn_sz,heads, d_dim)
        self.att_self2 = MultiHeadselfAttention(Input_attn_sz,heads, d_dim)
        self.att_cross1 = MultiHeadcrossAttention(Input_attn_sz, Input_attn_sz, heads, d_dim)
        self.att_cross2 = MultiHeadcrossAttention(Input_attn_sz, Input_attn_sz, heads, d_dim)
        self.hb_pool = HybridPooling()
        
        self.lin_1 = nn.Linear(Input_attn_sz * 4, 256)
        self.lin_2 = nn.Linear(256, 256)
        self.lin_3 = nn.Linear(256, 1)
        self.pooling_drop = nn.Dropout(drop_in_pool)
        self.linear_drop = nn.Dropout(drop_in_linear)
        self.sig = nn.Sigmoid()
        self.relu1 = nn.ReLU()

    def forward(self, input_1, input_2, attention_mask_1, attention_mask_2):
        self.prot1_ot = self.conv_1(input_1)
        self.prot2_ot = self.conv_2(input_2)
        query_1s, key_1s, val_1s = self.prot1_ot, self.prot1_ot, self.prot1_ot
        query_2s, key_2s, val_2s = self.prot2_ot, self.prot2_ot, self.prot2_ot
        self.out_self_attn1 = self.att_self1(query_1s, key_1s, val_1s, attention_mask_1)
        self.out_self_attn2 = self.att_self2(query_2s, key_2s, val_2s, attention_mask_2)

        query_1c, key_2c, val_2c = self.prot1_ot, self.prot2_ot, self.prot2_ot
        query_2c, key_1c, val_1c = self.prot2_ot, self.prot1_ot, self.prot1_ot
        self.out_cross_attn1 = self.att_cross1(query_1c, key_2c, val_2c, attention_mask_2)
        self.out_cross_attn2 = self.att_cross2(query_2c, key_1c, val_1c, attention_mask_1)

        self.out_self_cross1 = (self.out_self_attn1 + self.out_cross_attn1)
        self.out_self_cross2 = (self.out_self_attn2 + self.out_cross_attn2)

        out_self_cross1 = self.pooling_drop(self.out_self_cross1)
        out_self_cross2 = self.pooling_drop(self.out_self_cross2)
        
        self.out_self_cross1 = self.hb_pool(out_self_cross1)
        self.out_self_cross2 = self.hb_pool(out_self_cross2)
        self.out = torch.cat((self.out_self_cross1, self.out_self_cross2), dim = 1)

        return self.sig(self.lin_3(self.linear_drop(self.relu1(self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out)))))))),self.prot2_ot,self.prot1_ot, self.out_self_cross1,self.out_self_cross2, self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out))))
    
##################################################################################################
def protein_matrix_mask(prot_seq_1, prot_seq_2, seq_encoding, seq_encoding_max_len, window, stride):
    protein_seq_1 = seq_encoding[prot_seq_1]
    protein_seq_2 = seq_encoding[prot_seq_2]
    protein_seq_1_len, protein_seq_2_len = len(protein_seq_1), len(protein_seq_2)
    protein_seq_1 = torch.nn.functional.pad(protein_seq_1, (0,0,0, seq_encoding_max_len - (protein_seq_1_len))).float()
    protein_seq_2 = torch.nn.functional.pad(protein_seq_2, (0,0,0, seq_encoding_max_len - (protein_seq_2_len))).float()
    prot_1_conv_mat = max(int((protein_seq_1_len - window)/stride) + 1, 1)
    prot_2_conv_mat = max(int((protein_seq_2_len - window)/stride) + 1, 1)
    conv_len_max = int((seq_encoding_max_len - window)/stride) + 1
    att_mask_prot1 = torch.cat((torch.full((prot_1_conv_mat, conv_len_max), 0).long(), torch.full((conv_len_max - prot_1_conv_mat, conv_len_max), 1).long())).long().transpose(-1, -2)
    att_mask_prot2 = torch.cat((torch.full((prot_2_conv_mat, conv_len_max), 0).long(), torch.full((conv_len_max - prot_2_conv_mat, conv_len_max), 1).long())).long().transpose(-1, -2)
    return protein_seq_1, protein_seq_2, att_mask_prot1.bool(), att_mask_prot2.bool()

class construct_dataset(data.Dataset):
    def __init__(self, data_sets, seq_encoding, seq_max_len = 1500, window = 20, stride = 10):
        super().__init__()
        self.prot_seq1 = data_sets["col1"].values.tolist()
        self.prot_seq2 = data_sets["col2"].values.tolist()
        self.y = np.array(data_sets["interaction"].values.tolist()).reshape([len(data_sets["interaction"]),1])
        
        self.seq_encoding = seq_encoding
        
        self.seq_max_len = seq_max_len
        self.window = window
        self.stride = stride

        
        
    def __len__(self):
        return len(self.y)
    
    
    
    def __getitem__(self, elements):
        Protein1, Protein2, attn_mask1, attn_mask2 = protein_matrix_mask(self.prot_seq1[elements], self.prot_seq2[elements], self.seq_encoding, seq_encoding_max_len = self.seq_max_len, window = self.window, stride = self.stride)
        
        return Protein1.to(device), Protein2.to(device), attn_mask1.to(device), attn_mask2.to(device), torch.tensor(self.y[elements], device=device, dtype=torch.float)

def tsne_comp(embeds,PCA_comp = 10,perplex=10):

    embeds = embeds.cpu().detach().numpy()

    from sklearn.decomposition import PCA

    # Reduce to 50 components first
    pca = PCA(n_components=PCA_comp, random_state=42)
    embeddings_pca = pca.fit_transform(embeds)
    embeddings_pca.shape
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplex, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_pca)
    return tsne_results


def  emb_of_raw(df):
    PN = df
    pro1 = []
    pro2 = []
    
    for i, row in PN.iterrows():
        
        pt1,pt2,a,b=protein_matrix_mask(row['col1'], row['col2'], mat_dict, seq_encoding_max_len = 1500, window = 20, stride = 10)
        pro1.append(pt1)
        pro2.append(pt2)


    protein_1_embeds = torch.stack(pro1)
    protein_2_embeds = torch.stack(pro2)

    protein_2_embeds.shape

    combined_embeds =torch.cat((protein_1_embeds, protein_2_embeds), axis=1)
    combined_embeds.shape
    combined_embeds = combined_embeds.view(combined_embeds.size(0), -1)
    return combined_embeds
    
final_val_probs1=[]
final_val_labels1=[]

###############################################

class Attnseq_PPI_model():
    def __init__(self,enc_dict,max_epoch=1000, early_stop = 8):
        self.enc_dict = enc_dict
        self.max_epoch = max_epoch
        self.thresh = 0.5
        self.early_stop = early_stop

    def model_training(self, train_data_sets, val_data_sets,k_fold_number):
        train_data = construct_dataset(train_data_sets, seq_encoding = self.enc_dict, seq_max_len = 1500, window = 20, stride = 10)
        train_loader = DataLoader(dataset = train_data, batch_size = 26, shuffle=True)
        val_data = construct_dataset(val_data_sets, seq_encoding = self.enc_dict, seq_max_len = 1500, window = 20, stride = 10)
        val_loader = DataLoader(dataset = val_data, batch_size = 26, shuffle=True)
        
        self.model = Self_cross_attn().to(device)
        self.opt = optim.Adam(params = self.model.parameters(), lr = 0.0001, weight_decay=0.00001)
        self.criterion = torch.nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.95)
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/human_autoppi_256/fold_{}'.format(k_fold_number+1))


        max_met = 100
        early_stop_count = 0
        
        for epoch in range(self.max_epoch):
            training_losses, validation_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []
            con1a,con2a, slf_cr1a, slf_cr2a, lin2a = [], [], [], [], []
            self.model.train()
            for i, (protein_1, protein_2, attention_mask_1, attention_mask_2, labels) in enumerate(train_loader):
                self.opt.zero_grad()
                probs = self.model(protein_1, protein_2, attention_mask_1, attention_mask_2)
                probs,con1,con2, slf_cr1, slf_cr2, lin2 = self.model(protein_1, protein_2, attention_mask_1, attention_mask_2)
                
                loss = self.criterion(probs, labels)
                loss.backward()
                self.opt.step()
                training_losses.append(loss)
                train_probs.extend(probs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                train_labels.extend(labels.cpu().clone().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
            loss_epoch = self.criterion(torch.tensor(train_probs).float(), torch.tensor(train_labels).float())
            ###########
            self.writer.add_scalar('Loss/train', (loss_epoch), epoch+1)
            ###########
            print("===========================================", flush = True)
            print("===========================================", flush = True)
            print("training loss:: " + str(loss_epoch), flush = True)

            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](train_labels, train_probs, thresh = self.thresh)
                if(key == "auc"):
                    metrics = metrics_dict[key](train_labels, train_probs)
                    self.writer.add_scalar('AUC/train',(metrics), epoch+1)
                    
                else:
                    metrics = metrics_dict[key](train_labels, train_probs)
                    self.writer.add_scalar('AUPR/train',(metrics), epoch+1)
                    
                    
                print("train_" + key + ": " + str(metrics), flush=True)

            tn_tr, fp_tr, fn_tr, tp_tr = conf_matrix(train_labels, train_probs, thresh = self.thresh)
            print("train_true_negative:: value: %f, epoch: %d" % (tn_tr, epoch + 1), flush=True)
            print("train_false_positive:: value: %f, epoch: %d" % (fp_tr, epoch + 1), flush=True)
            print("train_false_negative:: value: %f, epoch: %d" % (fn_tr, epoch + 1), flush=True)
            print("train_true_positive:: value: %f, epoch: %d" % (tp_tr, epoch + 1), flush=True)
            self.scheduler.step()
            print("------------------------------------------", flush = True)
            print("------------------------------------------", flush = True)

            self.model.eval()
            for i, (protein_1, protein_2, attention_mask_1, attention_mask_2, labels) in enumerate(val_loader):
                with torch.no_grad():
                    probs,con11,con22, slf_cr11, slf_cr22, lin22 = self.model(protein_1, protein_2, attention_mask_1, attention_mask_2)

                    loss = self.criterion(probs, labels)

                    validation_losses.append(loss)
                    val_probs.extend(probs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    val_labels.extend(labels.cpu().detach().squeeze(1).numpy().astype('int32').flatten().tolist())
                    
                    
                    con1a.extend(con11.cpu().detach())
                    con2a.extend(con22.cpu().detach())
                    slf_cr1a.extend(slf_cr11.cpu().detach())
                    slf_cr2a.extend(slf_cr22.cpu().detach())
                    lin2a.extend(lin22.cpu().detach())

            val_probs1 = torch.tensor(val_probs)
            
            val_probs = (val_probs1.squeeze() > 0.5).int()
            # Compute Confusion Matrix
            cm = confusion_matrix(val_labels, val_probs)
            tn, fp, fn, tp = cm.ravel()

            labels = []
            for i in range(len(val_labels)):
                if val_labels[i] == 1 and val_probs[i] == 1:
                    labels.append("TP")
                elif val_labels[i] == 0 and val_probs[i] == 0:
                    labels.append("TN")
                elif val_labels[i] == 0 and val_probs[i] == 1:
                    labels.append("FP")
                elif val_labels[i] == 1 and val_probs[i] == 0:
                    labels.append("FN")

            # Define Colors
            color_map = {"TP": "green", "TN": "blue", "FP": "red", "FN": "orange"}

            # loss_epoch = self.criterion(torch.tensor(val_probs).float(), torch.tensor(val_labels).float())
            loss_epoch = self.criterion(val_probs1.float(), torch.tensor(val_labels).float())
            ##############
            self.writer.add_scalar('Loss/test', (loss_epoch), epoch+1)
            ################
            print("validation loss:: "+ str(loss_epoch), flush = True)
            
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    metrics = metrics_dict[key](val_labels, val_probs, thresh = self.thresh)
                    
                if(key == "auc"):
                    metrics = metrics_dict[key](train_labels, train_probs)
                    self.writer.add_scalar('AUC/val',(metrics), epoch+1)
                else:
                    metrics = metrics_dict[key](val_labels, val_probs)
                    self.writer.add_scalar('AUPR/val',(metrics), epoch+1)
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
                    
                    try:
                        ########################
                        raw = emb_of_raw(df=b4)
                        ####################
                        con1a = torch.stack(con1a)
                        con2a = torch.stack(con2a)
                        con = torch.cat((con1a, con2a), dim = 1)
                        con = con.view(con.size(0), (con.size(1)*con.size(2)))
                        # print(con1a.shape)
                        # con2a.shape
                        ################
                        slf_cr1a = torch.stack(slf_cr1a)
                        slf_cr2a = torch.stack(slf_cr2a)
                        # print(slf_cr1a.shape)
                        # slf_cr22.shape
                        cr = torch.cat((slf_cr1a, slf_cr2a), dim = 1)
                        ################################################
                        lin2a = torch.stack(lin2a)

                        print(lin2a.shape)

                        # plot_tsne(PN=b4, combined_embeds=raw, con=con, cr=cr, lin2=lin2a,fold=k_fold_number)
                        
                        
                        
                        # Apply t-SNE for Each Hidden Layer
                        # =======================================
                        
                        intermediate_layers = [con,cr,lin2a]
                        plt.figure(figsize=(15, 10))

                        for i, layer_output in enumerate(intermediate_layers):
                            # layer_features = layer_output.numpy()
                            # tsne = TSNE(n_components=2, random_state=42)
                            # features_2D = tsne.fit_transform(layer_features)
                            
                            if i != 2:
                                tsne_results = tsne_comp(layer_output,PCA_comp=10)
                            else:
                                tsne_results = tsne_comp(layer_output,PCA_comp=10, perplex=10)
                            
                            
                            
                            # Plot t-SNE Representation
                            # plt.subplot(2, 2, i + 1)
                            # for label in set(labels):
                            #     indices = [j for j, l in enumerate(labels) if l == label]
                            #     plt.scatter(features_2D[indices, 0], features_2D[indices, 1], 
                            #                 label=label, color=color_map[label], alpha=0.7)
                            
                            
                            ### simple version code for above commented code
                            plt.subplot(2, 2, i + 2)
                            unique_labels = set(labels)
                            for label in unique_labels:
                                indices = []
                                for idx, current_label in enumerate(labels):
                                    if current_label == label:
                                        indices.append(idx)
                                        
                                plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                                            label=label, color=color_map[label], alpha=0.7)
                                
                                
                            ########################################   
                            
                            plt.title(f"t-SNE Visualization - Layer {i+1}")
                            plt.xlabel("t-SNE Dimension 1")
                            plt.ylabel("t-SNE Dimension 2")
                            plt.legend()
                            
                            
                            
                        tsne_results = tsne_comp(raw,PCA_comp=10)
                        y_test = torch.tensor(b4["interaction"].values, dtype=torch.float)
                        # Plot Raw Input Data
                        plt.subplot(2, 2, 1)
                        for label in np.unique(y_test.numpy()):
                            indices = np.where(y_test.numpy() == label)
                            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                                        label="interacting" if label == 1 else "Non-interacting", 
                                        color="green" if label == 1 else "red", alpha=0.6)
                                
                        
                        plt.suptitle("Feature Space Evolution Across Model Layers", fontsize=16)
                        plt.legend()
                        
                        plt.tight_layout()
                        plt.savefig(fr'../{k_fold_number}.png')
                        
                        #######none
                        tsne_results =None
                    except:
                        print('Error in tsne plot_RAM consumption', flush=True)
                    
                    #######
                    
                    
                    
                    
                    
                    break





        ## outside forloop ######

        
        
        print(f'Threshold value is {self.thresh}', flush=True)
        
        
        final_val_probs1.append(np.array(final_val_probs))
        final_val_labels1.append(np.array(final_val_labels))
        
        
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

#######################################################################################################
#######################################################################################################
    
PN = pd.read_csv( r"../Human_1.5k.csv")
PN0 = PN['col1']
PN1 = PN['col2']
interaction = PN['interaction']
kf = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
k_fold_number = 0
###############################################################################################################
def k_fold_crossval_model(train_data, val_data,k_fold_number):
    print("5 fold cross validation on human dataset............", flush = True)
    training_data = train_data
    validation_data = val_data

    net = Attnseq_PPI_model(mat_dict)
    out = net.model_training(training_data, validation_data,k_fold_number)

for  train1, test1 in kf.split(PN0, interaction):

  a1 = PN0[train1]
  a2 = PN1[train1]
  a3 = interaction[train1]

  a4= pd.concat((a1,a2,a3),axis=1)


  b1=  PN0[test1]
  b2 = PN1[test1]
  b3 = interaction[test1]
  b4= pd.concat((b1,b2,b3),axis=1)
  
  
  k_fold_crossval_model(train_data=a4, val_data=b4,k_fold_number=k_fold_number)

  print('===================================================')
  k_fold_number +=1
  print(f'Done k fold_{k_fold_number}')
  print('===================================================')

#####################################################################################################

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

df1.loc['mean_training'] = mean_values1
df.loc['mean_testing'] = mean_values

print('training\n')
print(df1)

print('validation\n')
print(df)

# # Save the DataFrame to a CSV file
df.to_csv(r'../Val_hu.csv')
df1.to_csv(r'../Train_hu.csv')




###########################################################################################
###########################################################################################

num_folds = 5

y_true_folds = final_val_labels1
y_score_folds = final_val_probs1

tprs = []
fprs = []
precisions = []
recalls = []
roc_aucs = []
pr_aucs = []

import sklearn.metrics as metrics
plt.figure(figsize=(12, 5))

# Subplot for ROC Curve
plt.subplot(1, 2, 1)
for i in range(num_folds):
    fpr, tpr, _ = metrics.roc_curve(y_true_folds[i], y_score_folds[i])
    roc_auc = metrics.auc(fpr, tpr)
    tprs.append(tpr)
    fprs.append(fpr)
    roc_aucs.append(roc_auc)
    plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC = {roc_auc:.2f})", alpha=0.7)

# Mean ROC Curve
mean_tpr = np.mean(np.array([np.interp(np.linspace(0, 1, 100), fprs[i], tprs[i]) for i in range(num_folds)]), axis=0)
mean_fpr = np.linspace(0, 1, 100)
mean_auc =  metrics.auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label=f"Mean ROC (AUC = {mean_auc:.2f})", lw=2)
plt.plot([0, 1], [0, 1], 'r--', lw=1)  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (5-Fold CV)")
plt.legend()
# plt.show()

# Subplot for PR Curve
plt.subplot(1, 2, 2)
for i in range(num_folds):
    precision, recall, _ = metrics.precision_recall_curve(y_true_folds[i], y_score_folds[i])
    pr_auc =  metrics.auc(recall, precision)
    precisions.append(precision)
    recalls.append(recall)
    pr_aucs.append(pr_auc)
    plt.plot(recall, precision, label=f"Fold {i+1} (AP = {pr_auc:.2f})", alpha=0.7)

# Mean PR Curve
mean_precision = np.mean(np.array([np.interp(np.linspace(0, 1, 100), recalls[i][::-1], precisions[i][::-1]) for i in range(num_folds)]), axis=0)
mean_recall = np.linspace(0, 1, 100)
mean_pr_auc =  metrics.auc(mean_recall, mean_precision)

plt.plot(mean_recall, mean_precision, 'k--', label=f"Mean PR (AP = {mean_pr_auc:.2f})", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (5-Fold CV)")
plt.legend()

plt.tight_layout()
# plt.show()


plt.savefig(r'../roc_auc_pr_256.png')



###########################################################################################
###########################################################################################




































































# ############################################################################################


# ############################################################################################

# # import numpy as np
# # import pandas as pd
# # import torch
# # import matplotlib.pyplot as plt
# # from sklearn.manifold import TSNE

# # embeddings_dict = r"C:\Users\dipay\OneDrive\Documents\EMBEDDING\yeast\mat_dict_yeast_core.pth"

# # embeddings_dict = torch.load(embeddings_dict)

# # PN = pd.read_csv( r"K:\My Drive\PPI_paper_REVISED\DATASETS\yeast_core\NP_concat_new.csv")
# # # PN = pd.read_csv( r"K:\My Drive\PPI_paper_REVISED\Datasets_&_w2v_models\yeast_core\NP_concat_new_shuffeled.csv")
# # # PN = pd.read_csv( r"/scratch/chiranjibs.nbu/Human_data_1.5k/Human_1.5k.csv")
# # # PN = pd.read_csv( r"/scratch/chiranjibs.nbu/yeast_core/NP_concat_new.csv")
# # PN11 = PN[:200]
# # PN12 = PN[10342:]
# # PN= pd.concat([PN11,PN12], ignore_index=True)

# # pro1 = []
# # pro2 = []
# # ms11= []
# # ms22 = []


# # for i, row in PN.iterrows(): 
    
# #     # print(i)
# #     # print(row['col1'])

# #     pt1,pt2,ms1,ms2=protein_matrix_mask(row['col1'], row['col2'], embeddings_dict, seq_encoding_max_len = 1500, window = 20, stride = 10)
# #     pro1.append(pt1)
# #     pro2.append(pt2)
# #     ms11.append(ms1)
# #     ms22.append(ms2)


# # protein_1_embeds = torch.stack(pro1)
# # protein_2_embeds = torch.stack(pro2)

# # protein_2_embeds.shape

# # combined_embeds =torch.cat((protein_1_embeds, protein_2_embeds), axis=1)
# # combined_embeds.shape
# # combined_embeds = combined_embeds.view(combined_embeds.size(0), -1)






# # # def  emb_of_raw(df):
    
# # #     PN = df
# # #     pro1 = []
# # #     pro2 = []
    
# # #     for i, row in PN.iterrows(): 
        
# # #         pt1,pt2=protein_matrix_mask(row['col1'], row['col2'], embeddings_dict, seq_encoding_max_len = 1500, window = 20, stride = 10)
# # #         pro1.append(pt1)
# # #         pro2.append(pt2)


# # #     protein_1_embeds = torch.stack(pro1)
# # #     protein_2_embeds = torch.stack(pro2)

# # #     protein_2_embeds.shape

# # #     combined_embeds =torch.cat((protein_1_embeds, protein_2_embeds), axis=1)
# # #     combined_embeds.shape
# # #     combined_embeds = combined_embeds.view(combined_embeds.size(0), -1)
# # #     return combined_embeds
    
        
        













# # combined_embeds = combined_embeds.numpy()

# from sklearn.decomposition import PCA

# # Reduce to 50 components first
# pca = PCA(n_components=50, random_state=42)
# embeddings_pca = pca.fit_transform(combined_embeds)
# embeddings_pca.shape

# # combined_embeds.view(400,3000*1024)
# # Apply t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_results = tsne.fit_transform(embeddings_pca)

# # Plot t-SNE
# plt.figure(figsize=(10, 7))
# scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
# plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.title("t-SNE Projection of Raw Protein Embeddings")
# plt.show()

# ############################################################################################
# ############################################################################################


# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# embeddings_dict = r"C:\Users\dipay\OneDrive\Documents\EMBEDDING\yeast\mat_dict_yeast_core.pth"

# embeddings_dict = torch.load(embeddings_dict)

# PN = pd.read_csv( r"K:\My Drive\PPI_paper_REVISED\DATASETS\yeast_core\NP_concat_new.csv")
# # PN = pd.read_csv( r"K:\My Drive\PPI_paper_REVISED\Datasets_&_w2v_models\yeast_core\NP_concat_new_shuffeled.csv")
# # PN = pd.read_csv( r"/scratch/chiranjibs.nbu/Human_data_1.5k/Human_1.5k.csv")
# # PN = pd.read_csv( r"/scratch/chiranjibs.nbu/yeast_core/NP_concat_new.csv")
# PN11 = PN[:200]
# PN12 = PN[10342:]
# PN= pd.concat([PN11,PN12], ignore_index=True)

# pro1 = []
# pro2 = []
# ms11= []
# ms22 = []


# for i, row in PN.iterrows(): 
    
#     # print(i)
#     # print(row['col1'])

#     pt1,pt2,ms1,ms2=protein_matrix_mask(row['col1'], row['col2'], embeddings_dict, seq_encoding_max_len = 1500, window = 20, stride = 10)
#     pro1.append(pt1)
#     pro2.append(pt2)
#     ms11.append(ms1)
#     ms22.append(ms2)
    
    
    

# pro1 = torch.stack(pro1)
# pro2 = torch.stack(pro2)

# pro1.shape, pro2.shape

# ms11 = torch.stack(ms11)
# ms22 = torch.stack(ms22)
# ms11.shape, ms22.shape



# class Self_cross_attn(nn.Module):
#     def __init__(self, Input_attn_sz = 100, kernel_sz = 20, stride = 10, heads = 4, d_dim = 32, conv_in_shape = 1024, drop_in_pool = 0.5, drop_in_linear = 0.3):
#         super(Self_cross_attn, self).__init__()
#         self.conv_1 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
#         self.conv_2 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
#         self.att_self1 = MultiHeadselfAttention(Input_attn_sz,heads, d_dim)
#         self.att_self2 = MultiHeadselfAttention(Input_attn_sz,heads, d_dim)
#         self.att_cross1 = MultiHeadcrossAttention(Input_attn_sz, Input_attn_sz, heads, d_dim)
#         self.att_cross2 = MultiHeadcrossAttention(Input_attn_sz, Input_attn_sz, heads, d_dim)
#         self.hb_pool = HybridPooling()
        
#         self.lin_1 = nn.Linear(Input_attn_sz * 4, 64)
#         self.lin_2 = nn.Linear(64, 16)
#         self.lin_3 = nn.Linear(16, 1)
#         self.pooling_drop = nn.Dropout(drop_in_pool)
#         self.linear_drop = nn.Dropout(drop_in_linear)
#         self.sig = nn.Sigmoid()
#         self.relu1 = nn.ReLU()

#     def forward(self, input_1, input_2, attention_mask_1, attention_mask_2):
#         self.prot1_ot = self.conv_1(input_1)
#         self.prot2_ot = self.conv_2(input_2)
#         query_1s, key_1s, val_1s = self.prot1_ot, self.prot1_ot, self.prot1_ot
#         query_2s, key_2s, val_2s = self.prot2_ot, self.prot2_ot, self.prot2_ot
#         self.out_self_attn1 = self.att_self1(query_1s, key_1s, val_1s, attention_mask_1)
#         self.out_self_attn2 = self.att_self2(query_2s, key_2s, val_2s, attention_mask_2)

#         query_1c, key_2c, val_2c = self.prot1_ot, self.prot2_ot, self.prot2_ot
#         query_2c, key_1c, val_1c = self.prot2_ot, self.prot1_ot, self.prot1_ot
#         self.out_cross_attn1 = self.att_cross1(query_1c, key_2c, val_2c, attention_mask_2)
#         self.out_cross_attn2 = self.att_cross2(query_2c, key_1c, val_1c, attention_mask_1)

#         self.out_self_cross1 = (self.out_self_attn1 + self.out_cross_attn1)
#         self.out_self_cross2 = (self.out_self_attn2 + self.out_cross_attn2)

#         out_self_cross1 = self.pooling_drop(self.out_self_cross1)
#         out_self_cross2 = self.pooling_drop(self.out_self_cross2)
        
#         self.out_self_cross1 = self.hb_pool(out_self_cross1)
#         self.out_self_cross2 = self.hb_pool(out_self_cross2)

#         # self.out_self_cross1, _ = torch.max(out_self_cross1, dim = 1)
#         # self.out_self_cross2, _ = torch.max(out_self_cross2, dim = 1)

#         self.out = torch.cat((self.out_self_cross1, self.out_self_cross2), dim = 1)

#         return self.sig(self.lin_3(self.linear_drop(self.relu1(self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out)))))))),self.prot2_ot,self.prot1_ot, self.out_self_cross1,self.out_self_cross2, self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out))))




# slf_cr= Self_cross_attn()

# # ot,con1, slf_cr1= slf_cr(pt1,pt2,ms1,ms2)
# ot,con1,con2, slf_cr1, slf_cr2, lin2= slf_cr(pro1,pro2,ms11,ms22)


# ####################
# # con1 = torch.stack(con1, dim=1).squeeze(1)

# con = torch.cat((con1, con2), dim = 1)
# con.shape
# con = con.view(con.size(0), (con.size(1)*con.size(2)))
# con.shape

# ################
# cr = torch.cat((slf_cr1, slf_cr2), dim = 1)
# cr.shape

# # slf_cr1[0].shape
# # slf_cr1.shape

# ###############
# lin2.shape
# # lin2 = lin2.view(lin2.size(0), (lin2.size(1)*lin2.size(2)))




# def tsne_comp(combined_embeds,PCA_comp = 50):

#     combined_embeds = combined_embeds.detach().numpy()

#     from sklearn.decomposition import PCA

#     # Reduce to 50 components first
#     pca = PCA(n_components=PCA_comp, random_state=42)
#     embeddings_pca = pca.fit_transform(combined_embeds)
#     embeddings_pca.shape

#     # combined_embeds.view(400,3000*1024)
#     # Apply t-SNE
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     tsne_results = tsne.fit_transform(embeddings_pca)
#     return tsne_results



# plt.figure(figsize=(15, 10))

# for i in range(1,5):
    
#     if i==1:
#         tsne_results = tsne_comp(combined_embeds,PCA_comp=50)
#         # Plot t-SNE
#         # plt.figure(figsize=(8, 8))
#         plt.subplot(2, 2, i)
#         scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#         plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#         plt.xlabel("t-SNE Dimension 1")
#         plt.ylabel("t-SNE Dimension 2")
#         plt.title("t-SNE Projection of Raw Protein Embeddings")
#         plt.legend()
        
#     elif i==2:
#         tsne_results = tsne_comp(con,PCA_comp=50)
#         # Plot t-SNE
#         # plt.figure(figsize=(8, 8))
#         plt.subplot(2, 2, 2)
#         scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#         plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#         plt.xlabel("t-SNE Dimension 1")
#         plt.ylabel("t-SNE Dimension 2")
#         plt.title("t-SNE Projection of convolution layer")
#         plt.legend()
    
#     elif i==3:
#         tsne_results = tsne_comp(cr,PCA_comp=50)
#         # Plot t-SNE
#         # plt.figure(figsize=(8, 8))
#         plt.subplot(2, 2, i)
#         scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#         plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#         plt.xlabel("t-SNE Dimension 1")
#         plt.ylabel("t-SNE Dimension 2")
#         plt.title("t-SNE Projection of self-cross attention")
#         plt.legend()
    
#     elif i==4:
#         tsne_results = tsne_comp(lin2,PCA_comp=lin2.shape[1])
#         # Plot t-SNE
#         # plt.figure(figsize=(8, 8))
#         plt.subplot(2, 2, i)
#         scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#         plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#         plt.xlabel("t-SNE Dimension 1")
#         plt.ylabel("t-SNE Dimension 2")
#         plt.title("t-SNE Projection of linear layer")
#         plt.legend()
        
        
# plt.show()

# ############################################################################################


# def plot_tsne(PN, combined_embeds, con, cr, lin2):

#     def tsne_comp(embeds,PCA_comp = 50):

#         embeds = embeds.detach().numpy()

#         from sklearn.decomposition import PCA

#         # Reduce to 50 components first
#         pca = PCA(n_components=PCA_comp, random_state=42)
#         embeddings_pca = pca.fit_transform(embeds)
#         embeddings_pca.shape

#         # combined_embeds.view(400,3000*1024)
#         # Apply t-SNE
#         tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#         tsne_results = tsne.fit_transform(embeddings_pca)
#         return tsne_results



#     plt.figure(figsize=(15, 10))

#     for i in range(1,5):
        
#         if i==1:
#             tsne_results = tsne_comp(combined_embeds,PCA_comp=50)
#             # Plot t-SNE
#             # plt.figure(figsize=(8, 8))
#             plt.subplot(2, 2, i)
#             scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#             plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#             plt.xlabel("t-SNE Dimension 1")
#             plt.ylabel("t-SNE Dimension 2")
#             plt.title("t-SNE Projection of Raw Protein Embeddings")
#             plt.legend()
            
#         elif i==2:
#             tsne_results = tsne_comp(con,PCA_comp=50)
#             # Plot t-SNE
#             # plt.figure(figsize=(8, 8))
#             plt.subplot(2, 2, 2)
#             scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#             plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#             plt.xlabel("t-SNE Dimension 1")
#             plt.ylabel("t-SNE Dimension 2")
#             plt.title("t-SNE Projection of convolution layer")
#             plt.legend()
        
#         elif i==3:
#             tsne_results = tsne_comp(cr,PCA_comp=50)
#             # Plot t-SNE
#             # plt.figure(figsize=(8, 8))
#             plt.subplot(2, 2, i)
#             scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#             plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#             plt.xlabel("t-SNE Dimension 1")
#             plt.ylabel("t-SNE Dimension 2")
#             plt.title("t-SNE Projection of self-cross attention")
#             plt.legend()
        
#         elif i==4:
#             tsne_results = tsne_comp(lin2,PCA_comp=lin2.shape[1])
#             # Plot t-SNE
#             # plt.figure(figsize=(8, 8))
#             plt.subplot(2, 2, i)
#             scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=PN["interaction"], cmap="coolwarm", alpha=0.6)
#             plt.colorbar(scatter, label="Interaction (0 = No, 1 = Yes)")
#             plt.xlabel("t-SNE Dimension 1")
#             plt.ylabel("t-SNE Dimension 2")
#             plt.title("t-SNE Projection of linear layer")
#             plt.legend()
            
            
#     # plt.show()
#     plt.savefig(r'K:\My Drive\PPI_paper_revised_with_added_algorithms\epoch.png')




































