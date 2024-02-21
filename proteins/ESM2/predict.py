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
print(f"Target device is {device}, if specified..")
print('============================START============================')

#------------DEFS------------#

import torch
import pandas as pd
import numpy as np

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

def makefeatures(attention_tensor, contacts_tensor):
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

    X = whole_array[:, :-1]  # Features array (excluding last column)
    y = whole_array[:, -1]  # Target array (last column)

    return X, y

import seaborn as sns
import matplotlib.pyplot as plt

def plot_result(real, predictions, len, save_path_name, model, pdb, f1):
    real = real.reshape(len,len)
    predictions = predictions.reshape(len,len)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title(f'Structure: {pdb}')
    axs[0].set_xlabel("Sequence")
    axs[0].set_ylabel("Sequence")
    sns.heatmap(real, cmap='binary', ax=axs[0], vmin=0, vmax=1, cbar=False)
    axs[0].invert_yaxis()

    axs[1].set_title(f'{model} prediction - F1 score: {f1:.2f}')
    axs[1].set_xlabel("Sequence")
    axs[1].set_ylabel("Sequence")
    sns.heatmap(predictions, cmap='binary', ax=axs[1], vmin=0, vmax=1, cbar=False)
    axs[1].invert_yaxis()

    plt.tight_layout()

    # Save or Show the contact maps as a single PNG file
    plt.savefig(save_path_name)
    #plt.show()
    plt.close(fig)

print('Loading models..')

#-----------LOAD LM-----------#

mode = input('Choose: "8M", "35M", "150M", "650M", "3B" or "exit" : ')

if mode == '8M':
    
    from transformers import EsmTokenizer, EsmModel

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    lm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)

    num_layers = 6
    num_heads = 20

    logregpath = './logreg_esm2_8M.joblib'

elif mode == '35M':

    from transformers import EsmTokenizer, EsmModel

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    lm = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)

    num_layers = 12
    num_heads = 20

    logregpath = './logreg_esm2_35M.joblib'

elif mode == '150M':

    from transformers import EsmTokenizer, EsmModel

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    lm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device)

    num_layers = 30
    num_heads = 20

    logregpath = './logreg_esm2_150M.joblib'

elif mode == '650M':

    from transformers import EsmTokenizer, EsmModel

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    lm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    num_layers = 33
    num_heads = 20

    logregpath = './logreg_esm2_650M.joblib'

elif mode == '3B':

    from transformers import EsmTokenizer, EsmModel

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    lm = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D").to(device)

    num_layers = 36
    num_heads = 40

    logregpath = './logreg_esm2_3B.joblib'

elif mode == 'exit':
    import sys
    sys.exit()

else:
    import sys
    print('No such mode. Exiting.')
    sys.exit()

#---------LOAD LOGREG--------#

import joblib
from sklearn.linear_model import LogisticRegression

chosen_model = joblib.load(logregpath)
model = 'Logistic Regression'

print('Language model (', device, ') and Logistic Regression ( cpu ) modules loaded.\nProcessing test data..')

#--------------------------------------------------------------------------------------------

import os
import json
from tqdm import tqdm

os.makedirs(model, exist_ok=True)

#df = pd.DataFrame(columns=['Entry ID', 'F1 Score', 'Recall', 'Precision'])
df = pd.DataFrame(columns=['Entry ID', 'F1 Score'])

total_f1 = 0
total_recall = 0
total_precision = 0

included = 0
excluded = []

with open('../dataset/trRosetta.json') as f: # Give the test dataset
    proteins = json.load(f)

for protein in tqdm(proteins):
    proteinid = protein['PDBcode']
    sequence = protein['Sequence']
    length = len(sequence)

    try:
        if length < 1023:
            included += 1
            # Attention--------------------------------------------------------------------
            inputs = tokenizer(sequence, return_tensors='pt', max_length = 1024, truncation=True).to(device)
            # Pass the input data through the model and retrieve the attention weights
            outputs = lm(**inputs, output_attentions=True)
            attentions = outputs.attentions
            attention_weights = torch.tensor(remove_unwanted(attentions))

            # Structure--------------------------------------------------------------------
            for contact in os.listdir('../contacts/'): # Provide directory with nearby contacts (original)
                if contact[:8] == proteinid:
                    structure = np.load('../contacts/'+contact)['b']

            # MakeFeatures-----------------------------------------------------------------
            X , y_real = makefeatures(attention_weights, structure)

            y_pred = chosen_model.predict(X)
            y_pred = (y_pred >= 0.5)#.float().cpu()

            from sklearn.metrics import f1_score, recall_score, precision_score

            f1_score = f1_score(y_real, y_pred, average = 'macro')
            recall = recall_score(y_real, y_pred, zero_division=1)
            precision = precision_score(y_real, y_pred, zero_division=1)

            total_f1 += f1_score
            total_recall += recall
            total_precision += precision

            # save individual scores to df
            df.loc[len(df)] = [proteinid, f1_score]

            #save = f'./{model}/' + f'{proteinid}_{model}'
            #plot_result(y_real, y_pred, length, save, model, proteinid[:4], f1_score)

    except:
        excluded.append(proteinid)

#df.loc[len(df)] = ['average', total_f1/included, total_recall/included, total_precision/included]
df.to_csv(f'./bpreds/' + mode + '.csv', index=False)

print('Successfully predicted:', included)
print('Done. Bye!')

