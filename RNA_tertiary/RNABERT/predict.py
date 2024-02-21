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

def remove_padding(padded_attention, original_length):
    pad = 440 - original_length
    attention = padded_attention[:, :, :-pad, :-pad]
    return attention

import seaborn as sns
import matplotlib.pyplot as plt

def plot_result(real, predictions, len, save_path_name, model, pdb, f1):
    real = real.reshape(len,len)
    predictions = predictions.reshape(len,len)

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

#-------------------LOADMODEL

from rnabert import get_config, get_args, set_learned_params, BertModel, BertForMaskedLM, DATA

# Create a model instance
config = get_config("../../lms/RNABERT/RNA_bert_config.json")
config.hidden_size = config.num_attention_heads * config.multiple
args = get_args("../../lms/RNABERT/RNA_bert_args.json")
print('Config - Args: OK')

# Load model
bert_model = BertModel(config)
lmmodel = BertForMaskedLM(config, bert_model)
# Load the pretrained weights
pretrained = set_learned_params(lmmodel,'../../lms/RNABERT/bert_mul_2.pth')
pretrained.to(device)
print('RNABERT loaded in', device)

loader = DATA(args, config, device)

#----------------------------#

#---------LOAD MODEl---------#

import joblib

#N = input('Indicate smaple size : "20", "60", "140", "240", "all" : ')
mode = input('Choose: "logreg", "mlp", "tr", "rf", "xgb", "lda", "knn", "gnb" or "exit" : ')

for N in ["20", "60", "140", "240", "all"]:

    if mode == 'logreg':
        from sklearn.linear_model import LogisticRegression
        print('Running logistic regression. (cpu)')

        chosen_model = joblib.load(f"./models/logreg_rnabert_{N}.joblib")
        outname = f'logreg_{N}'
        model = 'Logistic Regression'

    elif mode == 'mlp':
        from sklearn.neural_network import MLPClassifier
        print('Running multi-layer perceptron. (cpu)')

        chosen_model = joblib.load(f'./models/mlp_rnabert_{N}.joblib')
        outname = f'mlp_{N}'
        model = 'Multi-Layer Perceptron'

    elif mode == 'tr':
        from sklearn import tree
        print('Running decision trees. (cpu)')

        chosen_model = joblib.load(f'./models/tr_rnabert_{N}.joblib')
        outname = f'tr_{N}'
        model = 'Decision Trees'

    elif mode == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        print('Running random forest. (cpu)')

        chosen_model = joblib.load(f'./models/rf_rnabert_{N}.joblib')
        outname = f'rf_{N}'
        model = 'Random Forest'

    elif mode == 'xgb':
        import xgboost as xgb
        print('Running xgboost classifier. (gpu)')

        chosen_model = joblib.load(f'./models/xgb_rnabert_{N}.joblib')
        outname = f'xgb_{N}'
        model = 'XGBoost'

    elif mode == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        print('Running linear discriminant analysis. (cpu)')

        chosen_model = joblib.load(f'./models/lda_rnabert_{N}.joblib')
        outname = f'lda_{N}'
        model = 'Linear Discriminant Analysis' 

    elif mode == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        print('Running k nearest neighbors. (cpu)')

        chosen_model = joblib.load(f'./models/knn_rnabert_{N}.joblib')
        outname = f'knn_{N}'
        model = 'K nearest neighbors'

    elif mode == 'gnb':
        from sklearn.naive_bayes import GaussianNB
        print('Running naive-bayes. (cpu)')

        chosen_model = joblib.load(f'./models/gnb_rnabert_{N}.joblib')
        outname = f'gnb_{N}'
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

    with open('../dataset/test_15%.json') as f:
        rnas = json.load(f)

    from rnabert import Infer

    count = 0

    for rna in tqdm(rnas):
        rnaid = rna['PDBcode']
        sequence = rna['Sequence']
        length = len(sequence)

        try:
            
            # Attention--------------------------------------------------------------------
            seqs, label, test_dl = loader.load_data_EMB(sequence) 
            outputer = Infer(config)
            padded_attention = outputer.revisit_attention(lmmodel, test_dl, seqs, attention_show_flg=True).squeeze(1)
            attention_weights = torch.tensor(remove_padding(padded_attention, length))


            # Secondary--------------------------------------------------------------------

            structure_1mer = np.load('../dataset/1mer_cmaps_contacts/'+rnaid+'_contacts.npz')['b'] # Input with nearby contacts

            # MakeFeatures-----------------------------------------------------------------
            X , y_real = makefeatures(attention_weights, structure_1mer)

            y_pred = chosen_model.predict(X)
            y_pred = (y_pred >= 0.5)#.float().cpu()

            from sklearn.metrics import f1_score, recall_score, precision_score

            f1_score = f1_score(y_real, y_pred, average = 'macro')
            recall = recall_score(y_real, y_pred, zero_division=1)
            precision = precision_score(y_real, y_pred, zero_division=1)

            total_f1 += f1_score
            total_recall += recall
            total_precision += precision

            count += 1

            # save individual scores to df
            #df.loc[len(df)] = [rnaid, f1_score, recall, precision]

            #save = './predictions/' + f'{rnaid}_{model}.pdf'
            #plot_result(y_real, y_pred, length, save, model, rnaid, f1_score)

        except:
            print(rnaid)

    df.loc[len(df)] = ['average', total_f1/count, total_recall/count, total_precision/count]
    df.to_csv(f'./wpreds/' + outname + '.csv', index=False)

