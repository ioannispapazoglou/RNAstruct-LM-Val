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

#-------------------LOADMODEL

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)

model = AutoModelForMaskedLM.from_pretrained("../../../lms/DNABERT/DNA_bert_3/",trust_remote_code=True, output_attentions=True)

model.to(device)
print('Running in',device,'..')

#----------------------------

import os
import json
import pandas as pd
from tqdm import tqdm

features = pd.DataFrame()
count = 0

with open(f'./training_3mer.json') as f:
    rnas = json.load(f)

for rna in tqdm(rnas):
    rnaid = rna['PDBcode']
    sequence = rna['Sequence']
    length = len(sequence)
    
    try:
        count += 1
        # Attention--------------------------------------------------------------------
        sequence_3mer = split_to_3mers(sequence)
        inputs = tokenizer(sequence_3mer, return_tensors='pt', max_length = 512, truncation=True).to(device)
        #inputs = {k: v.to(device) for k, v in inputs.items()}
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
        attention_weights = torch.tensor(attention_weights[:, :, 1:-1, 1:-1])

        # Secondary--------------------------------------------------------------------

        #structure_1mer = dot_bracket_to_matrix(structure)
        structure_3mer = np.load('../3mer_cmaps_nodiag_95/'+rnaid+'_contacts.npz')['b'] # Change input directory
        #structure_3mer = to_3mer(structure_1mer)#.numpy()

        # MakeFeatures-----------------------------------------------------------------
        X , y = makefeatures(attention_weights, structure_3mer, 4, undersampling=True)

        if count == 1:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    except:
       print(rnaid)

print('Samples:',count)
print('X:',X_train.shape,'y:', y_train.shape)
print('Saving..')

np.savez_compressed(f'./3mer_training.npz', x=X_train, y=y_train)


print('=============================END=============================')
