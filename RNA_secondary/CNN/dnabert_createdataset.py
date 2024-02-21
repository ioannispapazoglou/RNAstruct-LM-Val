import json 
import numpy as np
from tqdm import tqdm
import pickle
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

def split_to_3mers(string):
    mers = [string[i:i+3] for i in range(len(string)-2)]
    return ' '.join(mers)

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

def to_3mer(binary_map):
    # Get the dimensions of the input binary map
    height, width = binary_map.shape

    # Create an empty feature map with dimensions (initial-2, initial-2)
    feature_map = np.zeros((height - 2, width - 2), dtype=np.uint8)

    # Iterate over the binary map, excluding the boundary pixels
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 window from the binary map
            window = binary_map[i-1:i+2, j-1:j+2]

            # Check if the window contains at least one 1
            if np.sum(window) > 0:
                # Set the corresponding value in the feature map to 1
                feature_map[i-1, j-1] = 1

    return torch.tensor(feature_map, dtype=torch.float32)

def pad_structure(structure_maps, pad):
    
    i, j = structure_maps.shape
    padded_structure_maps = np.pad(structure_maps, ((0, pad - i), (0, pad - j)), mode='constant')
    padded_structure_maps = torch.tensor(padded_structure_maps, dtype=torch.float32)
    
    return padded_structure_maps

#-----------------ACTUALL_PROCESS:
print('==================================START==================================')
print('Creating dataset..')

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)

with open('./dataset/dnabert_dataset.json', 'r') as file: # Select input directory
    rnas = json.load(file)

inputs = [] # sequences
outputs = [] # structures
masks = []

for rna in tqdm(rnas):
    rnaid = rna['GeneralID']
    length = rna['Length']
    sequence = rna['Sequence'].replace('U','T') # convert RNA sequences to DNA sequences
    structure = rna['S_structure']

    sequence_3mer = split_to_3mers(sequence)
    lm_input = tokenizer(sequence_3mer, return_tensors='pt', padding='max_length', max_length = 512, truncation=True)
    inputs.append(lm_input['input_ids'])

    masks.append(lm_input['attention_mask'])

    structure_3mer = pad_structure(to_3mer(dot_bracket_to_matrix(structure)), 512).unsqueeze(0)
    outputs.append(structure_3mer)


all_input_tensors = torch.cat(inputs, dim=0)
all_output_tensors = torch.cat(outputs, dim=0)
all_mask_tensors = torch.cat(masks, dim=0)
#print(all_input_tensors.shape, all_output_tensors.shape, all_mask_tensors.shape)

dataset = TensorDataset(all_input_tensors, all_output_tensors, all_mask_tensors)

save_path = "./dataset.pkl" # Select output directory

# Save in hard drive
print('Saving..')
with open(save_path, "wb") as file:
    pickle.dump(dataset, file)

print('===================================END===================================')
