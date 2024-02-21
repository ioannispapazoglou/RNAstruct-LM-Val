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

def calculate_p(contactmap, attentionmap, th):
    
    l, h, i, j = attentionmap.shape
    numerator = np.zeros((l, h))
    denominator = np.zeros((l, h))

    # Apply threshold to attentionmap
    attentionmap_thresholded = np.where(attentionmap > th, 1, 0)
    
    attentionmap_mask = attentionmap > th

    for ll in range(l):
        for hh in range(h):
            numerator[ll][hh] = np.sum(contactmap * attentionmap_thresholded[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            denominator[ll][hh] = np.sum(attentionmap_thresholded[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            #print(numerator[ll][hh], denominator[ll][hh])
    return numerator, denominator

#-----------------------------------------------------------------------------

mode = input("Please choose mode from [uniref50-half, uniref50-xl, bfd, bfd-xl] : ")

print('Loading model..')

if mode == 'uniref50-half':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, :-1, :-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights
    
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', output_attentions=True)
    model = model.to(device)

    num_layers = 24
    num_heads = 32

elif mode == 'uniref50-xl':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, :-1, :-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50', output_attentions=True)
    model = model.to(device)

    num_layers = 24
    num_heads = 32

elif mode == 'bfd':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, 1:-1, 1:-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights

    from transformers import BertForMaskedLM, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd", output_attentions=True)
    model = model.to(device)

    num_layers = 30
    num_heads = 16

elif mode == 'bfd-xl':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, :-1, :-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights

    from transformers import T5Tokenizer, T5EncoderModel

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd', output_attentions=True)
    model = model.to(device)

    num_layers = 24
    num_heads = 32

else:
    print('No such model.. try again..')

print('Model successfully loaded and running in',device,'.')

#-----------------------------------------------------------------------------

print('Calculating molecule-wise propability..')
import os
import json
from tqdm import tqdm

with open('../dataset/trRosetta.json') as f:
    proteins = json.load(f)

grand_numerator = np.zeros((num_layers, num_heads))
grand_denominator = np.zeros((num_layers, num_heads))
probability = np.zeros((num_layers, num_heads))

ids_excluded = []
ids_memory = []
count = 0

for protein in tqdm(proteins):
    proteinid = protein['PDBcode']
    sequence = protein['Sequence']
    length = len(sequence)

    try:
    #if length <= 512:
        count += 1
        # Get attention
        elements = []
        for char in sequence:
            elements.append(char)

        seq = ''
        for element in elements:
            seq = seq + element + ' '

        token_encoding = tokenizer(seq, return_tensors='pt').to(device)
        
        with torch.no_grad():
            result = model(**token_encoding, output_attentions=True)

        attentions = result.attentions
        attention_weights = remove_unwanted(attentions)

        # Get structure (contacts)
        for contact in os.listdir('../dataset/pdb_nodiag/'): # Provide contact maps without nearby contacts
            if contact[:8] == proteinid:
                structure = np.load('../dataset/pdb_nodiag/'+contact)['b']

        # Check shape is the same
        if attention_weights[0][0].shape == structure.shape:
            # Calculate molecule P
            th = 0.3
            numerator, denominator = calculate_p(structure, attention_weights, th)
            grand_numerator += numerator
            grand_denominator += denominator
        
        else:
            ids_excluded.append(proteinid)
    except:
        ids_memory.append(proteinid)

    
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disgard parallelizing / fork errors

print('Molecule-wise calculation done: ',count,'total proteins.')
print('Any excluded ids because of memory needs:', len(ids_memory), ids_memory)
print('Any other excluded ids:', len(ids_excluded), ids_excluded)
print('Calculating summary probability on icluded...')
for l in range(num_layers):
    for h in range(num_heads):
        probability[l][h] = grand_numerator[l][h] / grand_denominator[l][h] if grand_denominator[l][h] != 0 else 0
probability = probability * 100
np.savez_compressed(f'./15k_prott5-p-{mode}-{th}-nodiag.npz', p=probability)

#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt

def heatdouble(heat,th):
    heat_2d = heat.reshape(num_layers, num_heads)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot heatmap on the first subplot
    im = ax1.imshow(heat_2d, cmap='Blues')
    ax1.invert_yaxis()
    ax1.set_title('15k | θ = ' + str(th))
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

    plt.savefig(f'./15051/512_15k_prott5-{mode}-{th}-nodiag.pdf', format='pdf')

heatdouble(probability, th)

print('All done! Bye.')

print('==========================END==========================')

