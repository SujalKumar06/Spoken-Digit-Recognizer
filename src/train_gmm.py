import pickle
import numpy as np
import os
from GMM import GMM

if not os.path.exists('train_data.pkl'):
    print("please run the feature_extraction.py first")
else:
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    models = {}
    for digit in range(10):
        d_str = str(digit)
        seqs = train_data[d_str]

        total_rows = sum(seq.shape[0] for seq in seqs)
        D = seqs[0].shape[1]

        X = np.zeros((total_rows, D))
        idx = 0
        for seq in seqs:
            rows = seq.shape[0]
            X[idx:idx + rows] = seq
            idx += rows

        print(f"Training GMM for Digit {digit}")

        gmm = GMM(n_gaussian=8, max_iter=100, tol=1e-4, n_restarts=3, reg_covar=1e-3)
        gmm.fit(X)
        models[d_str] = gmm

    with open('gmm_models.pkl', 'wb') as f:
        pickle.dump(models, f)

    print("GMM Training Over!")
