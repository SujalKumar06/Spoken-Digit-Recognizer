import pickle
import numpy as np
import os
from HMM import HMM
from utils import VectorQuantizer

if not os.path.exists('train_data.pkl'):
    print("please run the feature_extraction.py first")
else:
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    print("Training Vector Quantizer")
    all_frames = []
    for digit in range(10):
        d_str = str(digit)
        if d_str in train_data:
            for seq in train_data[d_str]:
                all_frames.append(seq)
                
    X_all = np.vstack(all_frames)
    vq = VectorQuantizer(n_symbols=64, max_iter=20)
    vq.fit(X_all)
    
    with open('vq_model.pkl', 'wb') as f:
        pickle.dump(vq, f)

    models = {}
    for digit in range(10):
        d_str = str(digit)
        seqs_continuous = train_data.get(d_str, [])

        if len(seqs_continuous) == 0:
            continue

        print(f"Training HMM for Digit {digit}")
        
        seqs_discrete = [vq.quantize(seq) for seq in seqs_continuous]

        hmm = HMM(n_states=5, n_symbols=64, max_iter=20, tol=1.0)
        hmm.fit(seqs_discrete)
        models[d_str] = hmm

    with open('hmm_models.pkl', 'wb') as f:
        pickle.dump(models, f)

    print("HMM Training Over!")