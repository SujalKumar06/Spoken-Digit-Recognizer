### Run the Code

1. `feature_extraction.py` - Will create `test_data.pkl` and `train_data.pkl`
2. `train_gmm.py` - Fits GMM on data, creates `gmm_models.pkl`
3. `train_hmm.py` - Fits HMM on data, creates `hmm_models.pkl`
4. `evaluate_gmm.py` - Evaluates the GMM models and gives the results
5. `evaluate_hmm.py` - Evaluates the HMM models and gives the results.
6. `compare.py` - Compares the two models and output results and saves 3 plots, `gmm_cm.py`, `hmm_cm.py`, `per_digit_comparision.py`
7. `inference.py` - It will ask you the path of the audio file, processing it, it will give results.
