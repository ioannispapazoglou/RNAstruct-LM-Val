import json 
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import TensorDataset
import copy
import math
import random 
import itertools

def kmer(seqs, k):
    kmer_seqs = []
    for seq in seqs:
        kmer_seq = []
        for i in range(len(seq)):
            if i <= len(seq)-k:
                kmer_seq.append(seq[i:i+k])
        kmer_seqs.append(kmer_seq)
    return kmer_seqs
            
def convert(seqs, kmer_dict, max_length):
    seq_num = [] 
    if not max_length:
        max_length = max([len(i) for i in seqs])
    for s in seqs:
        convered_seq = [kmer_dict[i] for i in s] + [0]*(max_length - len(s))
        seq_num.append(convered_seq)
    return seq_num

def make_dict(k):
    l = ["A", "U", "G", "C"]
    kmer_list = [''.join(v) for v in list(itertools.product(l, repeat=k))]
    kmer_list.insert(0, "MASK")
    dic = {kmer: i+1 for i,kmer in enumerate(kmer_list)}
    return dic

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

#---------------------------------------------------------------------------

def dot_bracket_to_matrix(dot_bracket):
    
    matrix = [[0] * len(dot_bracket) for _ in range(len(dot_bracket))]
    memory1 = []
    memory2 = []
    memory3 = []
    memory4 = []
    memory5 = []

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
        elif char == '{' :
            memory3.append(i)
        elif char == '}' :
            j = memory3.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == '<' :
            memory4.append(i)
        elif char == '>' :
            j = memory4.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == 'A' :
            memory5.append(i)
        elif char == 'a' :
            j = memory5.pop()
            matrix[j][i] = matrix[i][j] = 1

    adjacency_matrix = np.array(matrix)

    return adjacency_matrix

def pad_structure(structure_maps, pad):
    
    i, j = structure_maps.shape
    padded_structure_maps = np.pad(structure_maps, ((0, pad - i), (0, pad - j)), mode='constant')
    padded_structure_maps = torch.tensor(padded_structure_maps, dtype=torch.float32)
    
    return padded_structure_maps

def create_mask(length, max_length):
    mask = torch.zeros(max_length)
    mask[:length] = 1
    return mask

#-----------------ACTUALL_PROCESS:
print('==================================START==================================')
print('Creating dataset..')

with open('./dataset/rnabert_dataset.json', 'r') as file: # Select input directory
    rnas = json.load(file)

inputs = [] # sequences
outputs = [] # structures
masks = []

for rna in tqdm(rnas):
    rnaid = rna['GeneralID']
    length = rna['Length']
    sequence = rna['Sequence']
    structure = rna['S_structure']

    seqs=[]
    if set(sequence) <= set(['A', 'T', 'G', 'C', 'U']) and len(list(sequence)) < 440:
        seqs.append(sequence)
    
        kmer_seqs = kmer(seqs, k=1)
        masked_seq, low_seq = mask(kmer_seqs, rate=0, mag=1)
        kmer_dict = make_dict(1)
        low_seq = torch.tensor(convert(low_seq, kmer_dict, 440))
        inputs.append(low_seq)

        padded_structure_matrix = pad_structure(dot_bracket_to_matrix(structure), 440).unsqueeze(0)
        outputs.append(padded_structure_matrix)

        masks.append(create_mask(length, 440).unsqueeze(0))

all_input_tensors = torch.cat(inputs, dim=0)
all_output_tensors = torch.cat(outputs, dim=0)
all_mask_tensors = torch.cat(masks, dim=0)
print(all_input_tensors.shape, all_output_tensors.shape, all_mask_tensors.shape)

dataset = TensorDataset(all_input_tensors, all_output_tensors, all_mask_tensors)

save_path = "./dataset.pkl" # Select output directory

# Save in hard drive
print('Saving..')
with open(save_path, "wb") as file:
    pickle.dump(dataset, file)

print('===================================END===================================')
