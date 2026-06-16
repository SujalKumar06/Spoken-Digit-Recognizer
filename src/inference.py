import pickle
import numpy as np
from feature_extraction import extract_features

path = input("Enter Path for your audio file").strip()

features = extract_features(path)

gmm_models = pickle.load(open('gmm_models.pkl', 'rb'))
hmm_models = pickle.load(open('hmm_models.pkl', 'rb'))
vq_model   = pickle.load(open('vq_model.pkl', 'rb'))

best_gmm = -1
best_score = -np.inf

for d in range(10):
    d_str = str(d)
    if d_str in gmm_models:
        score = gmm_models[d_str].score(features)
        if score > best_score:
            best_score = score
            best_gmm = d

seq = vq_model.quantize(features)

best_hmm = -1
best_score = -np.inf

for d in range(10):
    d_str = str(d)
    if d_str in hmm_models:
        score = hmm_models[d_str].score(seq)
        if score > best_score:
            best_score = score
            best_hmm = d

print("GMM prediction:", best_gmm)
print("HMM prediction", best_hmm)
