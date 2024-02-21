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

def split_to_3mers(string):
    mers = [string[i:i+3] for i in range(len(string)-2)]
    return ' '.join(mers)

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

print('Loading model..')
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("../lms/DNABERT/DNA_bert_3/",trust_remote_code=True, output_attentions=True)
model.to(device)

print('Model successfully loaded and running in',device,'.')

#-----------------------------------------------------------------------------

print('Calculating molecule-wise propability..')
import json
from tqdm import tqdm

cutoff = '95'
with open('../dataset/redundunts.json') as f:
    rnas = json.load(f)

grand_numerator = np.zeros((12, 12))
grand_denominator = np.zeros((12, 12))
probability = np.zeros((12, 12))

ids_excluded = []

count = 0
for rna in tqdm(rnas):
    rnaid = rna['PDBcode']
    sequence = rna['Sequence']
    length = len(sequence)

    try:
        if length >= 10:
            # Save attention
            sequence_3mer = split_to_3mers(sequence)
            inputs = tokenizer(sequence_3mer, return_tensors='pt', max_length = 512, truncation=True).to(device)
            # Pass the input data through the model and retrieve the attention weights
            with torch.no_grad():
                outputs = model(**inputs)
                attentions = outputs.attentions
            attentions = tuple(att.detach().cpu().numpy() for att in attentions)
            
            attention_weights = []
            for layer in attentions:
                layer_attention_weights = []
                for head in layer:
                    layer_attention_weights.append(head)
                attention_weights.append(layer_attention_weights)
            attention_weights = np.squeeze(attention_weights, axis=1)
            attention_weights = attention_weights[:, :, 1:-1, 1:-1]

            # Save tertiary structure (contacts)
            structure_3mer = np.load(f'../dataset/3mer_cmaps_nodiag'+rnaid+'_contacts.npz')['b']
            #structure_1mer = add_diagonal_link(dot_bracket_to_matrix(structure), length)
            #structure_3mer = to_3mer(structure_1mer).numpy()
            
            # Check shape is the same
            if attention_weights[0][0].shape != structure_3mer.shape:
                ids_excluded.append(rnaid)

            # Calculate molecule P
            th = 0.3
            numerator, denominator = calculate_p(structure_3mer, attention_weights, th)
            grand_numerator += numerator
            grand_denominator += denominator

            count += 1

    except:
        print('Error on:', rnaid)
        print('Difference is:',int(attention_weights[0][0].shape[0]) - int(structure_3mer.shape[0]))
        print(attention_weights[0][0].shape,  structure_3mer.shape)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disgard parallelizing / fork errors

print('Molecule-wise calculation done!')
print('Any excluded ids:', len(ids_excluded), ids_excluded)
print('Calculating and saving summary probability on icluded...')
for l in range(12):
    for h in range(12):
        probability[l][h] = grand_numerator[l][h] / grand_denominator[l][h] if grand_denominator[l][h] != 0 else 0
probability = probability * 100
np.savez_compressed(f'dna_probability-{cutoff}-{th}.npz', p=probability)

#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt

def heatdouble(heat,th):
    heat_2d = heat.reshape(12,12)

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
    #plt.show()
    plt.savefig(f'./dna_3D-{cutoff}Å-{th}.pdf', format='pdf')

#heatdouble(probability, th)

print('Included :', count)
print('All done! Bye.')

print('==========================END==========================')

