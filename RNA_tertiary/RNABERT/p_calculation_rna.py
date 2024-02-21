print('=======================RESOURCES=======================')
import sys
import platform
import torch
import pandas as pd
import sklearn as sk

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")
print('=========================START=========================')

import numpy as np 

def remove_padding(padded_attention, original_length):
    pad = 440 - original_length
    attention = padded_attention[:, :, :-pad, :-pad]
    return attention

def dot_bracket_to_matrix(dot_bracket):
    
    matrix = [[0] * len(dot_bracket) for _ in range(len(dot_bracket))]
    memory1 = []
    memory2 = []

    for i, char in enumerate(dot_bracket):
        if char == '(' :
            memory1.append(i)
        elif char == ')' :
            j = memory1.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == '[' :
            memory2.append(i)
        elif char == ']' :
            j = memory2.pop()
            matrix[j][i] = matrix[i][j] = 1

    adjacency_matrix = np.array(matrix)

    return adjacency_matrix

def add_diagonal_link(adjacency_matrix, length):
    modified_matrix = np.copy(adjacency_matrix)

    for i in range(length - 1):
        modified_matrix[i, i + 1] = modified_matrix[i + 1, i] = 1

    return modified_matrix

def calculate_p(contactmap, attentionmap, th):
    
    l, h, i, j = attentionmap.shape
    numerator = np.zeros((l, h))
    denominator = np.zeros((l, h))
    
    attentionmap_mask = attentionmap > th

    for ll in range(l):
        for hh in range(h):
            numerator[ll][hh] = np.sum(contactmap * attentionmap[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            denominator[ll][hh] = np.sum(attentionmap[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            #print(numerator[ll][hh], denominator[ll][hh])
    return numerator, denominator

#-----------------------------------------------------------------------------

#Dependencies
import copy
import math
import json
from attrdict import AttrDict
import collections

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.multiprocessing as mp

import numpy as np
import random
import itertools
import os
import argparse
import datetime
import subprocess
from tqdm import tqdm
import time
import sys

from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans as KM
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering, SpectralClustering 

from Bio import SeqIO
from torchvision import transforms

#Get guidelines

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def get_config(file_path):
    config_file = file_path 
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)
    config = AttrDict(json_object)
    return config

def get_args(file_path):
    args_file = file_path  
    json_file = open(args_file, 'r')
    json_object = json.load(json_file)
    args = AttrDict(json_object)
    return args

#Architecture

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weight
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # bias
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.attention = BertAttention(config)

        self.intermediate = BertIntermediate(config)

        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs

        elif attention_show_flg == False:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output  # [batch_size, seq_length, hidden_size]

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        if attention_show_flg == True:
            self_output, attention_probs = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs

        elif attention_show_flg == False:
            self_output = self.selfattn(
                input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        # num_attention_heads: 12

        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads) 
        self.all_head_size = self.num_attention_heads * \
            self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
 
        if attention_show_flg == True:
            return context_layer, attention_probs
        elif attention_show_flg == False:
            return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])                        

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        all_encoder_layers = []
        all_attention_probs = []
        for i, layer_module in enumerate(self.layer):
            if attention_show_flg == True:
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg)
                all_attention_probs.append(attention_probs)
            elif attention_show_flg == False:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attention_show_flg == True:
            return all_encoder_layers, all_attention_probs
        elif attention_show_flg == False:
            return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)

        pooled_output = self.activation(pooled_output)

        return pooled_output

class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoded_layers, attention_probs = self.encoder(embedding_output,
                                                       extended_attention_mask,
                                                       output_all_encoded_layers, attention_show_flg)

        pooled_output = self.pooler(encoded_layers[-1])

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return encoded_layers, pooled_output, attention_probs



class BertPreTrainingHeads(nn.Module):
    def __init__(self, config ):
        super(BertPreTrainingHeads, self).__init__()

        self.predictions = MaskedWordPredictions(config)
        config.vocab_size = config.ss_size
        self.predictions_ss = MaskedWordPredictions(config)

        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        prediction_scores_ss = self.predictions_ss(sequence_output)

        seq_relationship_score = self.seq_relationship(
            pooled_output)

        return prediction_scores, prediction_scores_ss, seq_relationship_score

class MaskedWordPredictions(nn.Module):
    def __init__(self, config):
        super(MaskedWordPredictions, self).__init__()

        self.transform = BertPredictionHeadTransform(config)
        

        self.decoder = nn.Linear(in_features=config.hidden_size, 
                                 out_features=config.vocab_size,
                                 bias=False)
        self.bias = nn.Parameter(torch.zeros(
            config.vocab_size)) 

    def forward(self, hidden_states):
        hidden_states =self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.transform_act_fn = gelu

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class SeqRelationship(nn.Module):
    def __init__(self, config, out_features):
        super(SeqRelationship, self).__init__()

        self.seq_relationship = nn.Linear(config.hidden_size, out_features)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)

class BertForMaskedLM(nn.Module):
    def __init__(self, config, net_bert):
        super(BertForMaskedLM, self).__init__()

        self.bert = net_bert 

        self.cls = BertPreTrainingHeads(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention_show_flg=False):
        if attention_show_flg == False:
            encoded_layers, pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, attention_show_flg=False)
        else:
            encoded_layers, pooled_output, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, attention_show_flg=True)
            
        all_attention_probs = []
        for layer_attention_probs in attention_probs:
            all_attention_probs.append(layer_attention_probs)
            
        prediction_scores, prediction_scores_ss, seq_relationship_score = self.cls(encoded_layers[-1], pooled_output)
        return prediction_scores, prediction_scores_ss, encoded_layers, all_attention_probs


def set_learned_params(net, weights_path):
    loaded_state_dict = torch.load(weights_path, map_location=device)
    net.eval()
    param_names = []
    for name, param in net.named_parameters():
        param_names.append(name)
    new_state_dict = net.state_dict().copy()
    for index, (key_name, value) in enumerate(loaded_state_dict.items()):
        name = param_names[index]
        new_state_dict[name] = value 
        # print(str(key_name)+"→"+str(name))
        if (index+1 - len(param_names)) >= 0:
            break
    net.load_state_dict(new_state_dict)
    return net


def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.bert.encoder.layer[-1].named_parameters():
        param.requires_grad = True
    for name, param in model.bert.encoder.layer[-2].named_parameters():
        param.requires_grad = True
    for name, param in model.cls.named_parameters():
        param.requires_grad = True
    for name, param in model.bert.embeddings.named_parameters():
        param.requires_grad = False
    return model

#DataLoader

class DATA:
    def __init__(self, args, config, device):
        self.max_length = config.max_position_embeddings
        self.mag = args.mag
        self.maskrate = args.maskrate 
        self.batch_size = args.batch
        self.device = device

    def load_data_EMB(self, sequence):
        families = []
        gapped_seqs = []
        seqs = []

        gapped_seq = str(sequence).upper()
        gapped_seq = gapped_seq.replace("T", "U")
        seq = gapped_seq.replace('-', '')
        if set(seq) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(seq)) < self.max_length:
            seqs.append(seq)
            i = 0
            families.append(i)
            gapped_seqs.append(gapped_seq)
        gapped_seqs = np.tile(onehot_seq(gapped_seqs, self.max_length*5), (self.mag, 1))
        family = np.tile(np.array(families), self.mag)
        seqs_len = np.tile(np.array([len(i) for i in seqs]), self.mag)   
        k = 1   
        kmer_seqs = kmer(seqs, k)
        masked_seq, low_seq = mask(kmer_seqs, rate=0, mag=self.mag)
        kmer_dict = make_dict(k)
        swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
        masked_seq = np.array(convert(masked_seq, kmer_dict, self.max_length))
        low_seq = np.array(convert(low_seq, kmer_dict, self.max_length))

        # Move data to device
        gapped_seqs = torch.from_numpy(gapped_seqs).to(self.device)
        family = torch.from_numpy(family).to(self.device)
        seqs_len = torch.from_numpy(seqs_len).to(self.device)
        masked_seq = torch.from_numpy(masked_seq).to(self.device)
        low_seq = torch.from_numpy(low_seq).to(self.device)

        transform = transforms.Compose([transforms.ToTensor()])
        ds_MLM_SFP_ALIGN = MyDataset("SHOW", low_seq, masked_seq, family, seqs_len)
        dl_MLM_SFP_ALIGN = torch.utils.data.DataLoader(ds_MLM_SFP_ALIGN, self.batch_size, shuffle=False)
        return seqs, low_seq, dl_MLM_SFP_ALIGN

    
def base_to_num(seq, pad_max_length):
    seq = [list(i.translate(str.maketrans({'A': "2", 'U': "3", 'G': "4", 'C': "5"}))) for i in seq]
    seq = [list(map(lambda x : int(x), s)) for s in seq]
    seq = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in seq])
    return seq

def base_to_num(seq, pad_max_length):
    seq = [list(i.translate(str.maketrans({'A': "2", 'U': "3", 'G': "4", 'C': "5"}))) for i in seq]
    seq = [list(map(lambda x : int(x), s)) for s in seq]
    seq = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in seq])
    return seq

def num_to_base(seq):
    seq = seq.tolist()
    seq = ["".join(map(str, i)).replace('0', '').translate(str.maketrans({'2': "A", '3': "U", '4': "G", '5': "C"})) for i in seq]
    return seq

def mask_seq(seqs, rate = 0.2):
    c = np.random.rand(*seqs.shape)
    masked_seqs = np.where((c < 0.2) & (seqs != 0) , 1, seqs)
    d = np.random.randint(2, 6, c.shape)
    masked_seqs = np.where((c < 0.02) & (seqs != 0) , d, masked_seqs)
    return masked_seqs

def onehot_seq(gapped_seq, pad_max_length):
    gapped_seq = [list(i.translate(str.maketrans({'-': "0", '.' : "0", 'A': "1", 'U': "1", 'G': "1", 'C': "1"}))) for i in gapped_seq]
    gapped_seq = [list(map(lambda x : int(x), s)) for s in gapped_seq]
    gapped_seq = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in gapped_seq])
    return gapped_seq

def secondary_num(SS, pad_max_length):
    SS = [list(i.translate(str.maketrans({'.': "0", ':': "1", '<': "2", '>': "2", '(': "3", ')': "3", '{': "3", '}': "3", '[': "3", ']': "3", 'A': "4", 'a': "4", 'B': "4", 'b': "4", '-': "5", '_': "6", ',': "7"}))) for i in SS]
    SS = [list(map(lambda x : int(x), s)) for s in SS]
    SS = np.array([np.pad(s, ((0, pad_max_length-len(s)))) for s in SS])
    return SS

def kmer(seqs, k=1):
    kmer_seqs = []
    for seq in seqs:
        kmer_seq = []
        for i in range(len(seq)):
            if i <= len(seq)-k:
                kmer_seq.append(seq[i:i+k])
        kmer_seqs.append(kmer_seq)
    return kmer_seqs
            
def mask(seqs, rate = 0.2, mag = 1):
    seq = []
    masked_seq = []
    label = []
    for i in range(mag):
        seqs2 = copy.deepcopy(seqs)
        for s in seqs2:
            label.append(copy.copy(s))
            mask_num = int(len(s)*rate)
            all_change_index = np.array(random.sample(range(len(s)), mask_num))
            mask_index, base_change_index = np.split(all_change_index, [int(all_change_index.size * 0.90)])
            for i in list(mask_index):
                s[i] = "MASK"
            for i in list(base_change_index):
                s[i] = random.sample(('A', 'U', 'G', 'C'), 1)[0] 
            masked_seq.append(s)
    return masked_seq, label

def seq_label(seqs):
    return seqs

def convert(seqs, kmer_dict, max_length):
    seq_num = [] 
    if not max_length:
        max_length = max([len(i) for i in seqs])
    for s in seqs:
        convered_seq = [kmer_dict[i] for i in s] + [0]*(max_length - len(s))
        seq_num.append(convered_seq)
    return seq_num

def make_dict(k=3):
    l = ["A", "U", "G", "C"]
    kmer_list = [''.join(v) for v in list(itertools.product(l, repeat=k))]
    kmer_list.insert(0, "MASK")
    dic = {kmer: i+1 for i,kmer in enumerate(kmer_list)}
    return dic

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_type, low_seq, masked_seq, family, seq_len, low_seq_1 = None, masked_seq_1 = None, family_1 = None, seq_len_1 = None, common_index = None, common_index_1 = None, SS = None, SS_1 = None):
        self.train_type = train_type
        self.data_num = len(low_seq)
        self.low_seq = low_seq
        self.low_seq_1 = low_seq_1
        self.masked_seq = masked_seq
        self.masked_seq_1 = masked_seq_1
        self.family = family
        self.family_1 = family_1
        self.seq_len = seq_len
        self.seq_len_1 = seq_len_1
        self.common_index = common_index 
        self.common_index_1 = common_index_1 
        self.SS = SS
        self.SS_1 = SS_1
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_low_seq = self.low_seq[idx]
        out_masked_seq = self.masked_seq[idx]
        out_family = self.family[idx]
        out_seq_len = self.seq_len[idx]
        if self.train_type == "MLM" or self.train_type == "MUL" or self.train_type == "SSL":
            out_low_seq_1 = self.low_seq_1[idx]
            out_masked_seq_1 = self.masked_seq_1[idx]
            out_family_1 = self.family_1[idx]
            out_seq_len_1 = self.seq_len_1[idx]

        if self.train_type == "MUL" or self.train_type == "SSL":
            out_common_index = self.common_index[idx]
            out_common_index_1 = self.common_index_1[idx]

        if self.train_type == "SSL":
            out_SS = self.SS
            out_SS_1 = self.SS_1

        # if self.train_type == "SHOW":
        #     out_SS = self.SS

        if self.train_type == "MLM":
            return out_low_seq, out_masked_seq, out_family, out_seq_len, out_low_seq_1, out_masked_seq_1, out_family_1, out_seq_len_1
        elif self.train_type == "MUL":
            return out_low_seq, out_masked_seq, out_family, out_seq_len, out_low_seq_1, out_masked_seq_1, out_family_1, out_seq_len_1, out_common_index, out_common_index_1
        elif self.train_type == "SSL":
            return out_low_seq, out_masked_seq, out_family, out_seq_len, out_low_seq_1, out_masked_seq_1, out_family_1, out_seq_len_1, out_common_index, out_common_index_1, out_SS, out_SS_1
        # elif self.train_type == "SHOW":
        #     return out_low_seq, out_family, out_seq_len, out_SS
        else:
            return out_low_seq, out_family, out_seq_len

#Prediction

random.seed(10)
torch.manual_seed(1234)
np.random.seed(1234)
    
class Infer:
    
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def model_device(self, model):
        print("device: ", self.device)
        print('-----start-------')
        model.to(self.device)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return model

    def revisit_attention(self, model, dataloader, seqs, attention_show_flg=False):
        model.eval()
        torch.backends.cudnn.benchmark = True
        batch_size = dataloader.batch_size
        all_attention_maps = []
        for batch in dataloader:
            data, label, seq_len= batch
            inputs = data.to(self.device)
            prediction_scores, prediction_scores_ss, encoded_layers, attention_probs =  model(inputs, attention_show_flg=attention_show_flg)
            layer_attention_maps = []
            for i, layer_attention in enumerate(attention_probs):
                layer_attention_maps.append(layer_attention.detach().cpu().numpy())
            all_attention_maps.append(layer_attention_maps)
        all_attention_maps = np.concatenate(all_attention_maps, 1)

        if attention_show_flg:
            return all_attention_maps

#-----------------------------------------------------------------------------

print('Loading model..')

config = get_config("../lms/RNABERT/RNA_bert_config.json")
config.hidden_size = config.num_attention_heads * config.multiple
args = get_args("../lms/RNABERT/RNA_bert_args.json")
print('Config - Args: OK')

bert_model = BertModel(config)
model = BertForMaskedLM(config, bert_model)
pretrained = set_learned_params(model,'../lms/RNABERT/bert_mul_2.pth')
pretrained.to(device)

# Instanciate a dataloader
loader = DATA(args, config, device)

print('Model successfully loaded and running in',device,'.')

#-----------------------------------------------------------------------------

print('Calculating molecule-wise propability..')
import json
from tqdm import tqdm

cutoff = '95'
with open('../data/redundunts.json') as f:
    rnas = json.load(f)

grand_numerator = np.zeros((6, 12))
grand_denominator = np.zeros((6, 12))
probability = np.zeros((6, 12))

ids_excluded = []

count = 0

for rna in tqdm(rnas):
    rnaid = rna['PDBcode']
    sequence = rna['Sequence']
    length = len(sequence)

    try:
        if length >= 20:
            # Save attention
            seqs, label, test_dl = loader.load_data_EMB(sequence) 
            outputer = Infer(config)
            padded_attention = outputer.revisit_attention(model, test_dl, seqs, attention_show_flg=True).squeeze(1)
            attention_weights = remove_padding(padded_attention, length)

            # Save secondary structure (contacts)
            #structure_1mer = dot_bracket_to_matrix(structure)
            #structure_1mer = add_diagonal_link(dot_bracket_to_matrix(structure), length)
            structure_1mer = np.load(f'../dataset/1mer_cmaps_nodiag/pdb'+rnaid+'_contacts.npz')['b'] # Change input directory

            # Check shape is the same
            if attention_weights[0][0].shape != structure_1mer.shape:
                ids_excluded.append(rnaid)
            
            # Calculate molecule P
            th = 0.3
            numerator, denominator = calculate_p(structure_1mer, attention_weights, th)
            grand_numerator += numerator
            grand_denominator += denominator

            count += 1

    except:
        print('Error on:', rnaid)
        print('Difference is:',int(attention_weights[0][0].shape[0]) - int(structure_1mer.shape[0]))
        print(attention_weights[0][0].shape,  structure_1mer.shape)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disgard parallelizing / fork errors

print('Molecule-wise calculation done!')
print('Any excluded ids:', len(ids_excluded), ids_excluded)
print('Calculating summary probability on icluded...')
for l in range(6):
    for h in range(12):
        probability[l][h] = grand_numerator[l][h] / grand_denominator[l][h] if grand_denominator[l][h] != 0 else 0
probability = probability * 100
np.savez_compressed(f'rna_probability-{cutoff}-{th}.npz', p=probability)

#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt

def heatdouble(heat,th):
    heat_2d = heat.reshape(6,12)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot heatmap on the first subplot
    im = ax1.imshow(heat_2d, cmap='Blues', vmin = 0, vmax = 100)
    ax1.invert_yaxis()
    ax1.set_title('Th = ' + str(th))
    ax1.set_xlabel("Heads")
    ax1.set_ylabel("Layers")
    fig.colorbar(im, ax=ax1)

    # Plot vertical barplot on the second subplot
    max_values = np.max(heat_2d, axis=1)
    ax2.barh(np.arange(len(max_values)), max_values)
    ax2.set_title('Max Values')
    ax2.set_xlabel("Max Value")
    ax2.set_ylabel("Layer")
    ax2.set_yticks(np.arange(len(max_values)))
    ax2.set_yticklabels(np.arange(1, len(max_values)+1))

    plt.savefig(f'./math_rna/rna_3D-{cutoff}Å-{th}.pdf', format='pdf')

#heatdouble(probability, th)

print('Included: ', count)
print('All done! Bye.')

print('==========================END==========================')

