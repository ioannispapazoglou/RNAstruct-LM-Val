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

print('Loading training dataset..')

import numpy as np

N = input('Select interval : "20", "500", "2500", "5000", "8000", "14102" : ')

data = np.load(f'./dataset/batches/training_{N}.npz') # Select input directory
X = data['x']
y = data['y']


print('Training dataset:', X.shape, y.shape)


from joblib import dump

mode = input('Choose: "logreg", "mlp", "tr", "rf", "xgb", "lda", "knn", "gnb" : ')

if mode == 'logreg':
    from sklearn.linear_model import LogisticRegression
    print('Running logistic regression training.. (cpu)')

    logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
    logreg.fit(X, y)

    dump(logreg, f'./logreg_dnabert_{N}.joblib')

elif mode == 'mlp':
    from sklearn.neural_network import MLPClassifier
    print('Running multi-layer perceptron training.. (cpu)')

    mlp = MLPClassifier(max_iter=10000)
    mlp.fit(X, y)

    dump(mlp, f'./mlp_dnabert_{N}.joblib')

elif mode == 'tr':
    from sklearn import tree
    print('Running decision trees training.. (cpu)')

    tr = tree.DecisionTreeClassifier()
    tr.fit(X, y)

    dump(tr, f'./tr_dnabert_{N}.joblib')

elif mode == 'rf':
    from sklearn.ensemble import RandomForestClassifier
    print('Running random forest training.. (cpu)')

    rf = RandomForestClassifier()
    rf.fit(X, y)
    
    dump(rf, f'./rf_dnabert_{N}.joblib')

elif mode == 'xgb':
    import xgboost as xgb
    print('Running xgboost classifier training.. (gpu)')

    xgb = xgb.XGBClassifier(tree_method='gpu_hist')
    xgb.fit(X, y)

    dump(xgb, f'./xgb_dnabert_{N}.joblib')

elif mode == 'lda':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print('Running linear discriminant analysis training.. (cpu)')

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    dump(lda, f'./lda_dnabert_{N}.joblib')

elif mode == 'knn':
    from sklearn.neighbors import KNeighborsClassifier
    print('Running k nearest neighbors training.. (cpu)')

    knn = KNeighborsClassifier()
    knn.fit(X, y)

    dump(knn, f'./knn_dnabert_{N}.joblib')

elif mode == 'gnb':
    from sklearn.naive_bayes import GaussianNB
    print('Running naive-bayes training.. (cpu)')

    gnb = GaussianNB()
    gnb.fit(X, y)

    dump(gnb, f'./gnb_dnabert_{N}.joblib')

print('Training done. Saving..')


print('=============================END=============================')