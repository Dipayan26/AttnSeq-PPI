
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.simplefilter('ignore')
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
    def forward(self, Q, K, V, attn_mask):
        Q_spare, batch_size = Q, Q.size(0)
        
        q_s = self.Query_lin(Q).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        k_s = self.Key_lin(K).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        self.v_s = self.Value_lin(V).view(batch_size, -1, self.heads, self.dim).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, self.v_s)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim)
        context = self.out_dense(context)
        
        return context + Q_spare



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
        
    def forward(self, Q, K, V, attn_mask):
        Q_spare, batch_size = Q, Q.size(0)
        
        q_s = self.Query_lin(Q).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        k_s = self.Key_lin(K).view(batch_size, -1, self.heads, self.dim).transpose(1,2)
        self.v_s = self.Value_lin(V).view(batch_size, -1, self.heads, self.dim).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)

        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, self.v_s)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim)
        context = self.out_dense(context)
        
        return context + Q_spare
    
    
class conv_layers(nn.Module):
    def __init__(self, conv_in, conv_out, kernel_sz, stride, pooling = True, dropout = 0.5):
        super(conv_layers, self).__init__()
        self.pooling = pooling
        self.cnn = nn.Conv1d(conv_in, conv_out, kernel_size = kernel_sz, stride = stride)
     
        self.pool = torch.nn.MaxPool1d(3, stride = 1, padding=1)
        self.relu_func = nn.ReLU() 
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, mat):
        mat = torch.transpose(mat, -1, -2)
        mat = self.cnn(mat)
        mat = self.relu_func(mat)
        mat = self.dropout_layer(mat)

        mat = self.pool(mat) 
        mat = torch.transpose(mat, -1, -2)
        
        return mat

################################
class only_self_attn(nn.Module):
    def __init__(self, Input_attn_sz = 100, kernel_sz = 20, stride = 10, heads = 4, d_dim = 32, conv_in_shape = 128, drop_in_pool = 0.5, drop_in_linear = 0.3):
        super(only_self_attn, self).__init__()
        
        self.conv_1 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.conv_2 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
    
        self.att_self1 = MultiHeadselfAttention(Input_attn_sz,heads, d_dim)   
        self.att_self2 = MultiHeadselfAttention(Input_attn_sz,heads, d_dim)  
        self.lin_1 = nn.Linear(Input_attn_sz * 2, 64)
        self.lin_2 = nn.Linear(64, 16)
        self.lin_3 = nn.Linear(16, 1)

       
        self.pooling_drop = nn.Dropout(drop_in_pool)
        self.linear_drop = nn.Dropout(drop_in_linear)
        self.sig = nn.Sigmoid()
        self.relu1 = nn.ReLU()   
 
    def forward(self, input_1, input_2, attn_mask_1, attn_mask_2):
        self.prot1_ot = self.conv_1(input_1)
        self.prot2_ot = self.conv_2(input_2)
        query_1s, key_1s, val_1s = self.prot1_ot, self.prot1_ot, self.prot1_ot
        query_2s, key_2s, val_2s = self.prot2_ot, self.prot2_ot, self.prot2_ot
        self.out_self_attn1 = self.att_self1(query_1s, key_1s, val_1s, attn_mask_1)
        self.out_self_attn2 = self.att_self2(query_2s, key_2s, val_2s, attn_mask_2)
        out_self_cross1 = self.pooling_drop(self.out_self_attn1)
        out_self_cross2 = self.pooling_drop(self.out_self_attn2)

        self.out_self_cross1, _ = torch.max(out_self_cross1, dim = 1)
        self.out_self_cross2, _ = torch.max(out_self_cross2, dim = 1)
        
        self.out = torch.cat((self.out_self_cross1, self.out_self_cross2), dim = 1)

        return self.sig(self.lin_3(self.linear_drop(self.relu1(self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out))))))))

#############################################################
class only_cross_attn(nn.Module):
    def __init__(self, Input_attn_sz = 100, kernel_sz = 20, stride = 10, heads = 4, d_dim = 32, conv_in_shape = 128, drop_in_pool = 0.5, drop_in_linear = 0.3):
        super(only_cross_attn, self).__init__()
        self.conv_1 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.conv_2 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.att_cross1 = MultiHeadcrossAttention(Input_attn_sz, Input_attn_sz, heads, d_dim)   
        self.att_cross2 = MultiHeadcrossAttention(Input_attn_sz, Input_attn_sz, heads, d_dim)  
        self.lin_1 = nn.Linear(Input_attn_sz * 2, 64)
        self.lin_2 = nn.Linear(64, 16)
        self.lin_3 = nn.Linear(16, 1)
        self.pooling_drop = nn.Dropout(drop_in_pool)
        self.linear_drop = nn.Dropout(drop_in_linear)
        self.sig = nn.Sigmoid()
        self.relu1 = nn.ReLU()   
 
    def forward(self, input_1, input_2, attn_mask_1, attn_mask_2):
        self.prot1_ot = self.conv_1(input_1)
        self.prot2_ot = self.conv_2(input_2)
        query_1c, key_2c, val_2c = self.prot1_ot, self.prot2_ot, self.prot2_ot
        query_2c, key_1c, val_1c = self.prot2_ot, self.prot1_ot, self.prot1_ot
        self.out_cross_attn1 = self.att_cross1(query_1c, key_2c, val_2c, attn_mask_2)
        self.out_cross_attn2 = self.att_cross2(query_2c, key_1c, val_1c, attn_mask_1)
        out_self_cross1 = self.pooling_drop(self.out_cross_attn1)
        out_self_cross2 = self.pooling_drop(self.out_cross_attn2)
        self.out_self_cross1, _ = torch.max(out_self_cross1, dim = 1)
        self.out_self_cross2, _ = torch.max(out_self_cross2, dim = 1)
        self.out = torch.cat((self.out_self_cross1, self.out_self_cross2), dim = 1)
        return self.sig(self.lin_3(self.linear_drop(self.relu1(self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out))))))))

########################################################################
class NO_attention(nn.Module):
    def __init__(self, Input_attn_sz = 100, kernel_sz = 20, stride = 10, heads = 4, d_dim = 32, conv_in_shape = 128, drop_in_pool = 0.5, drop_in_linear = 0.3):
        super(NO_attention, self).__init__() 
        self.conv_1 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.conv_2 = conv_layers(conv_in = conv_in_shape, conv_out = Input_attn_sz, kernel_sz = kernel_sz, stride = stride, dropout = drop_in_pool)
        self.lin_1 = nn.Linear(Input_attn_sz * 2, 64)
        self.lin_2 = nn.Linear(64, 16)
        self.lin_3 = nn.Linear(16, 1)
        self.pooling_drop = nn.Dropout(drop_in_pool)
        self.linear_drop = nn.Dropout(drop_in_linear)
        self.sig = nn.Sigmoid()
        self.relu1 = nn.ReLU()   
 
    def forward(self, input_1, input_2, attn_mask_1, attn_mask_2):

        self.prot1_ot = self.conv_1(input_1)
        self.prot2_ot = self.conv_2(input_2)
        out_self_cross1 = self.pooling_drop(self.prot1_ot)
        out_self_cross2 = self.pooling_drop(self.prot2_ot)
        self.out_self_cross1, _ = torch.max(out_self_cross1, dim = 1)
        self.out_self_cross2, _ = torch.max(out_self_cross2, dim = 1)
        self.out = torch.cat((self.out_self_cross1, self.out_self_cross2), dim = 1)

        return self.sig(self.lin_3(self.linear_drop(self.relu1(self.lin_2(self.linear_drop(self.relu1(self.lin_1(self.out))))))))



