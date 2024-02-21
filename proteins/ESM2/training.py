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

#-----------------LOAD DATASET-----------------#
import numpy as np

mode = input('Choose: "8M", "35M", "150M", "650M", "3B" or "exit" : ')
print('Loading training dataset..')

if mode == '8M':
    
    data = np.load('./20_tr_8M_under.npz') 
    X = data['x']
    y = data['y']

elif mode == '35M':

    data = np.load('./20_tr_35M_under.npz') 
    X = data['x']
    y = data['y']

elif mode == '150M':

    data = np.load('./20_tr_150M_under.npz') 
    X = data['x']
    y = data['y']

elif mode == '650M':

    data = np.load('./20_tr_650M_under.npz') 
    X = data['x']
    y = data['y']

elif mode == '3B':

    data = np.load('./20_tr_3B_under.npz') 
    X = data['x']
    y = data['y']

elif mode == 'exit':
    import sys
    sys.exit()

else:
    import sys
    print('No such mode. Exiting.')
    sys.exit()

print('Training dataset:', X.shape, y.shape)

#-----------------TRAIN MODEL-----------------#

from joblib import dump
from sklearn.linear_model import LogisticRegression
print('Running logistic regression training.. (cpu)')

ho = input('Choose: Hyperparameter tuning mode (True, False): ') # False if no hyperparameter

if ho:
    logreg = LogisticRegression(n_jobs=-1, max_iter = 10000)
else:
    logreg = LogisticRegression(n_jobs=-1, max_iter = 10000, penalty = 'l1', solver = 'saga', C = 5.662979902262142)

logreg.fit(X, y)

if ho:
    dump(logreg, f'logreg_esm2_{mode}_HO.joblib')
else:
    dump(logreg, f'logreg_esm2_{mode}.joblib')

print('Training done. Model saved.')
print('Done. Bye!')

print('=============================END=============================')
