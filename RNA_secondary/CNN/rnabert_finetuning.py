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
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm
import numpy as np
import random
import itertools
import os
import argparse
import datetime
import subprocess
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
from torchvision import transforms, datasets
from Bio import AlignIO

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

#Prediction

random.seed(10)
torch.manual_seed(1234)
np.random.seed(1234)
    
class Infer:
    
    def __init__(self, config):
        self.device = (torch.device("cpu") if torch.cuda.is_available() else "cpu")
    
    def model_device(self, model):
        print("device: ", self.device)
        print('-----start-------')
        model.to(self.device)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return model

    def revisit_attention(self, model, dataloader, attention_show_flg=False):
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
            return torch.tensor(all_attention_maps).permute(1, 0, 2, 3, 4)

#-----------------ACTUALL_PROCESS:
import sys
import platform
import torch
import sklearn as sk
import pandas as pd

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "cpu" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")
print('==================================START==================================')

# Create a model instance
config = get_config("./LangMod/RNA_bert_config.json")
config.hidden_size = config.num_attention_heads * config.multiple
args = get_args("./LangMod/RNA_bert_args.json")
print('Config - Args: OK')

# Load model
bert_model = BertModel(config)
model = BertForMaskedLM(config, bert_model)
# Load the pretrained weights
pretrained = set_learned_params(model,'./LangMod/bert_mul_2.pth')
pretrained.to(device)
print('Model loaded in', device)

# Load ready dataset
import pickle
print('Loading dataset..')

load_path = "./dataset.pkl" # Select input directory
with open(load_path, "rb") as file:
    dataset = pickle.load(file)

from torch.utils.data import TensorDataset, DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

from torch.utils.data import SubsetRandomSampler

train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_sampler = SubsetRandomSampler(range(train_size))
valid_sampler = SubsetRandomSampler(range(train_size, train_size + valid_size))
test_sampler = SubsetRandomSampler(range(train_size + valid_size, len(dataset)))

train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=4, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler)

print("Number of training batches:",len(train_loader),
    "\nNumber of valifation batches:", len(valid_loader),
    "\nNumber of test batches:", len(test_loader))


class CustomTokenClassifier(nn.Module):
    def __init__(self, bert, bertoutputer, length):
        super(CustomTokenClassifier, self).__init__()
        self.bert = bert
        self.bertoutputer = bertoutputer
        self.length = length
        
        self.cnn = nn.Sequential(
            nn.Conv2d(72, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (length//4) * (length//4), 128),
            nn.ReLU(),
            nn.Linear(128, length*length)
        )
        
    def forward(self, dataloader):
        # Pass input through the pre-trained model
        padded_attention = self.bertoutputer.revisit_attention(self.bert, dataloader, attention_show_flg=True)

        # Reshape attention maps
        b, l, h, i, j = padded_attention.shape
        attention_maps = padded_attention.reshape(b, l*h, i, j)
       
        # Pass attention maps through CNN
        logits = self.cnn(attention_maps).reshape(-1, i, j)
        
        return logits


def remove_padding(binary_mask, padded_map):
    binary_mask = binary_mask.bool()  # Convert to boolean (T / F)
    
    # Calculate the indices of the unpadded region
    indices = torch.nonzero(binary_mask)
    
    # Get the minimum and maximum indices along each dimension
    min_indices, _ = torch.min(indices, dim=0)
    max_indices, _ = torch.max(indices, dim=0)
    
    # Extract the unpadded region from the padded map
    unpadded_map = padded_map[
        min_indices[0]:max_indices[0] + 1,
        min_indices[1]:max_indices[1] + 1
    ]
    
    return unpadded_map

def weigth_calculator(unpadded_target):
    
    n_samples = unpadded_target.shape[0]
    n_classes = 2

    weights = n_samples / (n_classes * torch.bincount(unpadded_target.reshape(-1).round().int()))
    if weights.shape == torch.Size([1]):
        weight = torch.ones([1])
    else: 
        weight = weights[1]
    return weight # they represent: [0, 1]

print('Training..')

length = 440
outputer = Infer(config)
custom_model = CustomTokenClassifier(model, outputer, length)
custom_model = custom_model.to(device)


# Define your optimizer
optimizer = AdamW(custom_model.parameters(), lr=2e-5)

# Define your training parameters
num_epochs = 10
patience = 3
best_loss = float('inf')
num_epochs_without_improvement = 0

# Training loop
for epoch in range(num_epochs):
    custom_model.train()
    
            
    loss = 0.0
    for batch in tqdm(train_loader):
        batch_loss = 0.0

        # Make new dataset to be imported in architecture ... 
        input_slot = batch[0].to(torch.long).to(device)
        label_slot = batch[1].to(device)
        mask_slot = batch[2].to(device)
        #print(input_slot.dtype, label_slot.dtype, mask_slot.dtype)
        dataset = TensorDataset(input_slot, label_slot, mask_slot)
        temp_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_logits = custom_model(temp_dataloader)

        for i in range(len(batch[0])):

            inputs = batch[0][i].unsqueeze(0).to(device)  # Get the input tensors from the batch
            targets = batch[1][i].unsqueeze(0).to(device)  # Get the output tensors from the batch

            masks = batch[2][i].to(device)  # Get the mask tensors from the batch {those are for the lm}
            lm_masks = batch[2][i].unsqueeze(0).to(device)  # Get the mask tensors from the batch {those are for the lm}
            structure_masks = torch.mm(masks.unsqueeze(1), masks.unsqueeze(0)).to(device)
            #print(structure_masks.shape)
            
            logits = all_logits[i]
            #print(logits.shape)

            unpadded_targets = remove_padding(structure_masks, targets[0]).unsqueeze(0)  # Remove padding from the output tensors
            unpadded_logits = remove_padding(structure_masks, logits).unsqueeze(0)  # Remove padding from the logits
            #print(unpadded_targets.shape, unpadded_logits.shape)

            weight = weigth_calculator(unpadded_targets[0])
            
            criterion = nn.BCEWithLogitsLoss(pos_weight = weight).to(device)
            #print(weight)

            loss = criterion(unpadded_logits, unpadded_targets)
            batch_loss += loss.item()
        
        loss += (batch_loss / len(batch[0]))
        
    avg_loss = loss / len(train_loader)
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    # Evaluate on validation set
    custom_model.eval()
    with torch.no_grad():

        loss = 0.0
        for batch in tqdm(valid_loader):
            batch_loss = 0.0

            # Make new dataset to be imported in architecture ... 
            input_ids = batch[0].to(torch.long).to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)
            dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
            temp_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            all_logits = custom_model(temp_dataloader).to(device)

            for i in range(len(batch[0])):

                inputs = batch[0][i].unsqueeze(0).to(device)  # Get the input tensors from the batch
                targets = batch[1][i].unsqueeze(0).to(device)  # Get the output tensors from the batch

                masks = batch[2][i].to(device)  # Get the mask tensors from the batch {those are for the lm}
                lm_masks = batch[2][i].unsqueeze(0).to(device)  # Get the mask tensors from the batch {those are for the lm}
                structure_masks = torch.mm(masks.unsqueeze(1), masks.unsqueeze(0)).to(device)
                #print(structure_masks.shape)
                
                logits = all_logits[i]
                #print(logits.shape)

                unpadded_targets = remove_padding(structure_masks, targets[0]).unsqueeze(0)  # Remove padding from the output tensors
                unpadded_logits = remove_padding(structure_masks, logits).unsqueeze(0)  # Remove padding from the logits
                #print(unpadded_targets.shape, unpadded_logits.shape)

                weight = weigth_calculator(unpadded_targets[0])
                
                criterion = nn.BCEWithLogitsLoss(pos_weight = weight).to(device)
                #print(weight)

                loss = criterion(unpadded_logits, unpadded_targets)
                batch_loss += loss.item()
            
            loss += (batch_loss / len(batch[0]))
        
        avg_loss = loss / len(valid_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss}")

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        num_epochs_without_improvement = 0
        # Save the best model
        torch.save(model.state_dict(), 'besttuneX10.pt')
    else:
        num_epochs_without_improvement += 1
        if num_epochs_without_improvement >= patience:
            print("Early stopping! No improvement for", patience, "epochs.")
            break

print('Evaluating..')

from sklearn.metrics import f1_score

custom_model.eval()
with torch.no_grad():

    df = pd.DataFrame(columns=['F1 Score'])
    total_f1 = 0
    for batch in tqdm(test_loader):

        
        batch_f1 = 0
        # Make new dataset to be imported in architecture ... 
        input_ids = batch[0].to(torch.long).to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        dataset = TensorDataset(input_ids, token_type_ids, attention_mask)
        temp_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        all_logits = custom_model(temp_dataloader).to(device)

        for i in range(len(batch[0])):

            inputs = batch[0][i].unsqueeze(0).to(device)  # Get the input tensors from the batch
            targets = batch[1][i].unsqueeze(0).to(device)  # Get the output tensors from the batch

            masks = batch[2][i].to(device)  # Get the mask tensors from the batch {those are for the lm}
            lm_masks = batch[2][i].unsqueeze(0).to(device)  # Get the mask tensors from the batch {those are for the lm}
            structure_masks = torch.mm(masks.unsqueeze(1), masks.unsqueeze(0)).to(device)
            #print(structure_masks.shape)
            
            logits = all_logits[i]
            #print(logits.shape)

            unpadded_targets = remove_padding(structure_masks, targets[0]).unsqueeze(0)  # Remove padding from the output tensors
            unpadded_logits = remove_padding(structure_masks, logits).unsqueeze(0)  # Remove padding from the logits
            #print(unpadded_targets.shape, unpadded_logits.shape)
        
            threshold = 0.5
            unpadded_predictions = (unpadded_logits > 0.5).float()

            unpadded_targets = unpadded_targets.squeeze(0).cpu()
            unpadded_predictions = unpadded_predictions.squeeze(0).cpu()
            
            f1 = f1_score(unpadded_targets, unpadded_predictions, average='macro', zero_division=1)
            batch_f1 += f1
        
        total_f1 += (batch_f1 / len(batch[0]))
        df.loc[len(df)] = [total_f1]

    avg_f1 = total_f1 / len(test_loader)
    df.to_csv(f'./CNN_ft_dna_X10.csv', index=False)

print(f"Average F1 score: {avg_f1}")

print('===================================END===================================')
