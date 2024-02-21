print('==========================RESOURCES==========================')
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
print('============================START============================')

import torch
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def remove_unwanted(attentions):
    attention_weights = []
    for layer in attentions:
        layer_attention_weights = []
        for head in layer:
            layer_attention_weights.append(head.detach().cpu().numpy())
        attention_weights.append(layer_attention_weights)
    attention_weights = np.squeeze(attention_weights, axis=1)
    attention_weights = attention_weights[:, :, 1:-1, 1:-1] # remove cls and sep tokens
    
    return attention_weights

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def preprocess(attention):
    attention = symmetrize(attention)
    attention = apc(attention)
    return attention

def makefeatures(attention_tensor, contacts_tensor, localcontacts, undersampling):
    #-------ATTENTION
    #----------------------PREPROCESS--------------------------------------------------------------------
    preprocessed_attn = preprocess(attention_tensor) # OR ANY OTHER PREPROCESSING STEP TO BE INSERTED HERE
    #----------------------------------------------------------------------------------------------------
    tensor_shape = preprocessed_attn.shape
    num_samples = tensor_shape[2]*tensor_shape[3]
    transposed_tensor = preprocessed_attn.permute(2, 3, 0, 1)
    attention_array = transposed_tensor.reshape((num_samples, tensor_shape[0]*tensor_shape[1]))

    #-------SECONDARYCONTACTS
    tensor_shape = contacts_tensor.shape
    reshaped_tensor = contacts_tensor.reshape((tensor_shape[0]*tensor_shape[1],))
    contacts_array = reshaped_tensor

    # Combine attention and contacts arrays
    whole_array = np.concatenate((attention_array, contacts_array.reshape(-1, 1)), axis=1)

    # Filter local contacts
    ilist = np.repeat(np.arange(contacts_tensor.shape[0]), contacts_tensor.shape[0])
    jlist = np.tile(np.arange(contacts_tensor.shape[0]), contacts_tensor.shape[0])
    mask = np.abs(ilist - jlist) > localcontacts
    filtered_array = whole_array[mask]

    X = whole_array[:, :-1]  # Features array (excluding last column)
    y = whole_array[:, -1]  # Target array (last column)

    if undersampling:
        under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        X_resampled, y_resampled = under_sampler.fit_resample(X, y) # Create final dataset balanced

        return X_resampled, y_resampled
    
    else:
        return X, y

#-------------------LOADMODEL

mode = input("Please choose mode from [8M , 35M , 150M , 650M, 3B] : ")

print('Loading model..')

from transformers import EsmTokenizer, EsmModel

if mode == '8M':
    
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = model.to(device)

    num_layers = 6
    num_heads = 20

elif mode == '35M':

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = model.to(device)

    num_layers = 12
    num_heads = 20

elif mode == '150M':

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
    model = model.to(device)

    num_layers = 30
    num_heads = 20

elif mode == '650M':

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)

    num_layers = 33
    num_heads = 20

elif mode == '3B':

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
    model = model.to(device)

    num_layers = 36
    num_heads = 40

else:
    print('No such model.. try again..')

print('Language Model successfully loaded and running in',device,'.')

#----------------------------

import os
import json
import pandas as pd
from tqdm import tqdm

undersampler = True

features = pd.DataFrame()
count = 0

with open('./trRosetta.json') as f:
    proteins = json.load(f)

for protein in tqdm(proteins):
    proteinid = protein['PDBcode']
    sequence = protein['Sequence']
    length = len(sequence)

    try:
        count += 1
        # Attention--------------------------------------------------------------------
        inputs = tokenizer(sequence, return_tensors='pt', max_length = 1024, truncation=True).to(device)
        # Pass the input data through the model and retrieve the attention weights
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions
        attention_weights = torch.tensor(remove_unwanted(attentions))

        # Structure--------------------------------------------------------------------
        for contact in os.listdir('../pdbs/'): #The original pdbs - nearby contacts are deleted in makefeatures def
            if contact[:4] == proteinid:
                structure = np.load('../pdbs/'+contact)['b']

        # MakeFeatures-----------------------------------------------------------------
        X , y = makefeatures(attention_weights, structure, 6, undersampling=undersampler) # Set nearby contacts range // undersampling mode

        if count == 1:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    except:
        print(proteinid)

print('Samples:',count)
print('X:',X_train.shape,'y:', y_train.shape)
print('Saving..')

if undersampler:
    np.savez_compressed(f'./20_tr_{mode}_under.npz', x=X_train, y=y_train)
else:
    np.savez_compressed(f'./20_tr_{mode}.npz', x=X_train, y=y_train)

print('=============================END=============================')
