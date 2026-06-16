import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plot_confusion_matrix

def load_all_data():
    files = ['train_data.pkl', 'test_data.pkl', 'gmm_models.pkl', 'hmm_models.pkl', 'vq_model.pkl']
    if not all(os.path.exists(f) for f in files):
        print("Run training scripts first.")
        return None
        
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open('gmm_models.pkl', 'rb') as f:
        gmm_models = pickle.load(f)
    with open('hmm_models.pkl', 'rb') as f:
        hmm_models = pickle.load(f)
    with open('vq_model.pkl', 'rb') as f:
        vq = pickle.load(f)
        
    return train_data, test_data, gmm_models, hmm_models, vq

def evaluate_model(models, data, model_type="GMM", vq_model=None):
    total_files = 0
    correct = 0
    cm = np.zeros((10, 10), dtype=int)
    
    for true_digit in range(10):
        true_str = str(true_digit)
        if true_str not in data:
            continue
            
        for feature_matrix in data[true_str]:
            total_files += 1
            
            if model_type == "HMM" and vq_model is not None:
                features = vq_model.quantize(feature_matrix)
            else:
                features = feature_matrix
                
            best_score = -np.inf
            predicted = -1
            
            for test_digit in range(10):
                test_str = str(test_digit)
                if test_str in models:
                    score = models[test_str].score(features)
                    if score > best_score:
                        best_score = score
                        predicted = test_digit
            
            cm[true_digit][predicted] += 1
            if predicted == true_digit:
                correct += 1
                
    accuracy = (correct / total_files) * 100 if total_files > 0 else 0
    
    per_digit = []
    for i in range(10):
        total_actual = np.sum(cm[i, :])
        if total_actual > 0:
            per_digit.append((cm[i][i] / total_actual) * 100)
        else:
            per_digit.append(0.0)
            
    return accuracy, cm, per_digit

def plot_per_digit_line_chart(classes, gmm_scores, hmm_scores):
    plt.figure(figsize=(10, 6))
    
    plt.plot(classes, gmm_scores, marker='o', linestyle='-', color='darkorange', label='GMM', linewidth=2, markersize=8)
    plt.plot(classes, hmm_scores, marker='s', linestyle='-', color='dodgerblue', label='HMM', linewidth=2, markersize=8)
    
    plt.title('Per-Digit Test Accuracy Comparison', fontsize=16, pad=15)
    plt.xlabel('Digit', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    for i in range(10):
        plt.text(i, gmm_scores[i] + 2.5, f"{gmm_scores[i]:.0f}%", ha='center', color='darkorange', fontweight='bold')
        plt.text(i, hmm_scores[i] - 4.5, f"{hmm_scores[i]:.0f}%", ha='center', color='dodgerblue', fontweight='bold')

    plt.tight_layout()
    filename = 'per_digit_comparison.png'
    plt.savefig(filename, dpi=300)
    plt.close()

assets = load_all_data()
if assets:
    train_data, test_data, gmm_models, hmm_models, vq = assets

    gmm_train_acc, _, _ = evaluate_model(gmm_models, train_data, "GMM")
    gmm_test_acc, gmm_cm, gmm_per_digit = evaluate_model(gmm_models, test_data, "GMM")

    hmm_train_acc, _, _ = evaluate_model(hmm_models, train_data, "HMM", vq)
    hmm_test_acc, hmm_cm, hmm_per_digit = evaluate_model(hmm_models, test_data, "HMM", vq)

    print(f"\nValidation score of GMM: {gmm_train_acc:.2f}%")
    print(f"Test score of GMM: {gmm_test_acc:.2f}%")
    print(f"Validation score of HMM: {hmm_train_acc:.2f}%")
    print(f"Test score of HMM: {hmm_test_acc:.2f}%\n")

    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(gmm_cm, classes, title='GMM Test Confusion Matrix', filename='gmm_cm.png')
    plot_confusion_matrix(hmm_cm, classes, title='HMM Test Confusion Matrix', filename='hmm_cm.png')

    plot_per_digit_line_chart(classes, gmm_per_digit, hmm_per_digit)
