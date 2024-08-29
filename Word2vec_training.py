import os
from Bio import SeqIO
from gensim.models import word2vec
import logging
###########################################################################################################
def import_fasta(filename):
    return [str(record.seq) for record in SeqIO.parse(filename, "fasta")]
def create_k_mer(seq, k_mer = 4):
    return [[seq[i][j:j + k_mer] for j in range(len(seq[i]) - k_mer + 1)] for i in range(len(seq))]
def w2v_model_train(dataset_path, model_out_path, k_mer = 4, vector_size = 128, window_size = 5, iteration = 100):
    seq_list = import_fasta(dataset_path)
    k_mers_list = create_k_mer(seq_list, k_mer = k_mer)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(k_mers_list, vector_size = vector_size, min_count = 1, window = window_size, epochs=iteration, sg = 1)
    os.makedirs(model_out_path, exist_ok = True)
    model.save(model_out_path + "/word2vec_model.pt")
    
############ Dataset and model out path ###################################################################
dataset_path = r""
model_out_path = r""
###########################################################################################################

w2v_model_train(dataset_path=dataset_path, model_out_path=model_out_path)