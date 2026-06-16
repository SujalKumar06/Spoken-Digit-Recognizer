import numpy as np
import matplotlib.pyplot as plt

def log_of_sum_exp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isneginf(a_max), 0.0, a_max)

    with np.errstate(divide='ignore'):
        sum_exp = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
        out = np.log(sum_exp) + a_max_safe

    if not keepdims:
        out = np.squeeze(out, axis=axis)
        a_max = np.squeeze(a_max, axis=axis)

    return np.where(np.isneginf(a_max), -np.inf, out)

class VectorQuantizer:
    def __init__(self, n_symbols=64, max_iter=20):
        self.n_symbols = n_symbols
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        n = X.shape[0]
        idx = np.random.choice(n, self.n_symbols, replace=False)
        self.centroids = X[idx].copy()

        for iteration in range(self.max_iter):
            D = np.stack([np.sum((X - c) ** 2, axis=1) for c in self.centroids])
            labels = np.argmin(D, axis=0)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_symbols):
                mask = (labels == k)
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.randint(n)]

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def quantize(self, X):
        D = np.stack([np.sum((X - c) ** 2, axis=1) for c in self.centroids])
        return np.argmin(D, axis=0)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', filename=None):
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')

    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.close()
