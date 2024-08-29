import torch
import torch.utils.data as data
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################################################################################################
def protein_matrix_mask(prot_seq_1, prot_seq_2, seq_encoding, seq_encoding_max_len, window, stride, k_mer):
    protein_seq_1 = seq_encoding[prot_seq_1]
    protein_seq_2 = seq_encoding[prot_seq_2]
    protein_seq_1_len, protein_seq_2_len = len(protein_seq_1), len(protein_seq_2)
    w2v_len_max = seq_encoding_max_len - k_mer + 1
    protein_seq_1 = torch.nn.functional.pad(protein_seq_1, (0,0,0, w2v_len_max - (protein_seq_1_len))).float()
    protein_seq_2 = torch.nn.functional.pad(protein_seq_2, (0,0,0, w2v_len_max - (protein_seq_2_len))).float()
    prot_1_conv_mat = max(int((protein_seq_1_len - window)/stride) + 1, 1)
    prot_2_conv_mat = max(int((protein_seq_2_len - window)/stride) + 1, 1)
    conv_len_max = int((w2v_len_max - window)/stride) + 1
    att_mask_prot1 = torch.cat((torch.full((prot_1_conv_mat, conv_len_max), 0).long(), torch.full((conv_len_max - prot_1_conv_mat, conv_len_max), 1).long())).long().transpose(-1, -2)
    att_mask_prot2 = torch.cat((torch.full((prot_2_conv_mat, conv_len_max), 0).long(), torch.full((conv_len_max - prot_2_conv_mat, conv_len_max), 1).long())).long().transpose(-1, -2)
    return protein_seq_1, protein_seq_2, att_mask_prot1.bool(), att_mask_prot2.bool()

class construct_dataset(data.Dataset):
    def __init__(self, data_sets, seq_encoding, seq_max_len = 1500, window = 20, stride = 10, k_mer = 4 ):
        super().__init__()
        self.prot_seq1 = data_sets["col1"].values.tolist()
        self.prot_seq2 = data_sets["col2"].values.tolist()
        self.y = np.array(data_sets["interaction"].values.tolist()).reshape([len(data_sets["interaction"]),1])
        self.seq_encoding = seq_encoding
        self.seq_max_len = seq_max_len
        self.window = window
        self.stride = stride
        self.k_mer = k_mer
    def __len__(self):
        return len(self.y)
    def __getitem__(self, elements):
        Protein1, Protein2, attn_mask1, attn_mask2 = protein_matrix_mask(self.prot_seq1[elements], self.prot_seq2[elements], self.seq_encoding, seq_encoding_max_len = self.seq_max_len, window = self.window, stride = self.stride, k_mer = self.k_mer)
        return Protein1.to(device), Protein2.to(device), attn_mask1.to(device), attn_mask2.to(device), torch.tensor(self.y[elements], device=device, dtype=torch.float)
