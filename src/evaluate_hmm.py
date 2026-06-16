import pickle
import numpy as np
import os

if not os.path.exists('test_data.pkl') or not os.path.exists('hmm_models.pkl') or not os.path.exists('vq_model.pkl'):
    print("run feature_extraction.py and train_hmm.py first")
else:
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
        
    with open('vq_model.pkl', 'rb') as f:
        vq = pickle.load(f)

    with open('hmm_models.pkl', 'rb') as f:
        models = pickle.load(f)

    total_files = 0
    correct = 0
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for true_digit in range(10):
        true_str = str(true_digit)
        if true_str not in test_data:
            continue

        for feature_matrix in test_data[true_str]:
            total_files += 1
            
            seq_discrete = vq.quantize(feature_matrix)
            
            best_score = -np.inf
            predicted = -1
            
            for test_digit in range(10):
                test_str = str(test_digit)
                if test_str in models:
                    score = models[test_str].score(seq_discrete)
                    if score > best_score:
                        best_score = score
                        predicted = test_digit
            
            confusion_matrix[true_digit][predicted] += 1
            if predicted == true_digit:
                correct += 1

    accuracy = (correct / total_files) * 100 if total_files > 0 else 0

    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print("\nPer-Digit Accuracy:")
    for i in range(10):
        total_actual = np.sum(confusion_matrix[i, :])
        if total_actual > 0:
            acc = (confusion_matrix[i][i] / total_actual) * 100
            print(f"  Digit {i}: {acc:>6.2f}% ({confusion_matrix[i][i]}/{total_actual})")
