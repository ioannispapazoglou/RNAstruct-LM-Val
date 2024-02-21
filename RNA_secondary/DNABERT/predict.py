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

def split_to_3mers(string):
    mers = [string[i:i+3] for i in range(len(string)-2)]
    return ' '.join(mers)

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

import seaborn as sns
import matplotlib.pyplot as plt

def plot_result(real, predictions, len, save_path_name, model, pdb, f1):
    real = real.reshape(len-2,len-2)
    predictions = predictions.reshape(len-2,len-2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title(f'Secondary Structure: {pdb}')
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

#-----------LOAD LM-----------#

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_3", trust_remote_code=True)

lmmodel = AutoModelForMaskedLM.from_pretrained("../lms/DNABERT/DNA_bert_3/",trust_remote_code=True, output_attentions=True)

lmmodel.to(device)
print('DNABERT loaded and running in',device,'..')

#---------LOAD MODEl---------#

import joblib

N = input('Select interval : "20", "500", "2500", "5000", "8000", "14102" : ')
mode = input('Choose: "logreg", "mlp", "tr", "rf", "xgb", "lda", "knn", "gnb" or "exit" : ')
 
print(f'========={mode}=={N}=========')

if mode == 'logreg':
    from sklearn.linear_model import LogisticRegression
    print('Running logistic regression. (cpu)')

    chosen_model = joblib.load(f"./logreg_dnabert_{N}.joblib")
    outname = f'logreg{N}'
    model = 'Logistic Regression'

elif mode == 'mlp':
    from sklearn.neural_network import MLPClassifier
    print('Running multi-layer perceptron. (cpu)')

    chosen_model = joblib.load(f'./mlp_dnabert_{N}.joblib')
    outname = f'mlp{N}'
    model = 'Multi-Layer Perceptron'

elif mode == 'tr':
    from sklearn import tree
    print('Running decision trees. (cpu)')

    chosen_model = joblib.load(f'./tr_dnabert_{N}.joblib')
    outname = f'tr{N}'
    model = 'Decision Trees'

elif mode == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    print('Running random forest. (cpu)')

    chosen_model = joblib.load(f'./rf_dnabert_{N}.joblib')
    outname = f'rf{N}'
    model = 'Random Forest'

elif mode == 'xgb':
    import xgboost as xgb
    print('Running xgboost classifier. (gpu)')

    chosen_model = joblib.load(f'./xgb_dnabert_{N}.joblib')
    outname = f'xgb{N}'
    model = 'XGBoost'

elif mode == 'lda':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print('Running linear discriminant analysis. (cpu)')

    chosen_model = joblib.load(f'./lda_dnabert_{N}.joblib')
    outname = f'lda{N}'
    model = 'Linear Discriminant Analysis' 

elif mode == 'knn':
    from sklearn.neighbors import KNeighborsClassifier
    print('Running k nearest neighbors. (cpu)')

    chosen_model = joblib.load(f'./knn_dnabert_{N}.joblib')
    outname = f'knn{N}'
    model = 'K nearest neighbors'

elif mode == 'gnb':
    from sklearn.naive_bayes import GaussianNB
    print('Running naive-bayes. (cpu)')

    chosen_model = joblib.load(f'./gnb_dnabert_{N}.joblib')
    outname = f'gnb{N}'
    model = 'Naive-Bayes'

elif mode == 'exit':
    import sys
    sys.exit()

else:
    import sys
    print('No such mode. Exiting.')
    sys.exit()

print('ML model loaded. Loading test data..')

import os
import json
from tqdm import tqdm

#os.makedirs(model, exist_ok=True)

df = pd.DataFrame(columns=['Entry ID', 'F1 Score', 'Recall', 'Precision'])

total_f1 = 0
total_recall = 0
total_precision = 0

with open('./dataset/test_15%.json') as f: # Select input directory
    rnas = json.load(f)

for rna in tqdm(rnas):
    rnaid = rna['GeneralID']
    length = rna['Length']
    sequence = rna['Sequence']
    structure = rna['S_structure']

    try:
        # Attention--------------------------------------------------------------------
        sequence_3mer = split_to_3mers(sequence)
        inputs = tokenizer(sequence_3mer, return_tensors='pt', max_length = 512, truncation=True).to(device)
        # Pass the input data through the model and retrieve the attention weights
        with torch.no_grad():
            outputs = lmmodel(**inputs)
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

        structure_1mer = dot_bracket_to_matrix(structure)
        structure_3mer = to_3mer(structure_1mer)#.numpy()

        # MakeFeatures-----------------------------------------------------------------
        X , y_real = makefeatures(attention_weights, structure_3mer)

        y_pred = chosen_model.predict(X)
        y_pred = (y_pred >= 0.5)#.float().cpu()

        from sklearn.metrics import f1_score, recall_score, precision_score

        f1_score = f1_score(y_real, y_pred, average='macro')
        recall = recall_score(y_real, y_pred, zero_division=1)
        precision = precision_score(y_real, y_pred, zero_division=1)

        total_f1 += f1_score
        total_recall += recall
        total_precision += precision

        # save individual scores to df
        #df.loc[len(df)] = [rnaid, f1_score, recall, precision]

        #save = f'./{model}/' + f'{rnaid}_{model}'
        #plot_result(y_real, y_pred, length, save, model, rnaid, f1_score)
    except:
        print(rnaid)

df.loc[len(df)] = ['average', total_f1/len(rnas), total_recall/len(rnas), total_precision/len(rnas)]
df.to_csv(f'./wpreds/' + outname + '.csv', index=False)

print('=================END=OF=PROGRAM=================')
