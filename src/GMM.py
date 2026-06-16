import numpy as np
from utils import log_of_sum_exp

class GMM:
    def __init__(self, n_gaussian = 8, max_iter = 100, tol = 1e-4,
                 n_restarts = 3, reg_covar = 1e-3):
        self.n_gaussian = n_gaussian
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.reg_covar = reg_covar
        self.weights = None
        self.means = None
        self.covars = None

    def _kmeans_centre(self, X):
        n = X.shape[0]
        idx = np.random.randint(n)
        centers = [X[idx].copy()]

        for i in range(1, self.n_gaussian):
            D_stack = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers])
            min_dist = np.min(D_stack, axis=0)

            probs = min_dist/(min_dist.sum() + 1e-12)
            idx = np.random.choice(n, p=probs)
            centers.append(X[idx].copy())

        return np.array(centers)

    def _kmeans_run(self, X, iters = 20):
        centers = self._kmeans_centre(X)
        for iter in range(iters):
            D_stack = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers])
            labels = np.argmin(D_stack, axis=0)
            
            for k in range(self.n_gaussian):
                cluster_points = []
                for i in range(len(labels)):
                    if labels[i] == k:
                        cluster_points.append(X[i])
                if len(cluster_points) > 0:
                    centers[k] = np.mean(cluster_points, axis=0)
            
        D_stack = np.stack([np.sum((X - c) ** 2, axis=1) for c in centers])
        labels = np.argmin(D_stack, axis=0)
    
        return centers, labels
    
    def _em_start(self, X):
        n, d = X.shape
        centers, labels = self._kmeans_run(X)

        self.means = centers.copy()
        self.weights = [0.0] * self.n_gaussian
        self.covars = [[0.0] * d for _ in range(self.n_gaussian)]

        for k in range(self.n_gaussian):
            cluster_points = []

            for i in range(n):
                if labels[i] == k:
                    cluster_points.append(X[i])

            cnt = len(cluster_points)

            self.weights[k] = max(cnt, 1)/n

            if cnt > 1:
                mean = [0.0] * d
                for point in cluster_points:
                    for j in range(d):
                        mean[j] += point[j]
                mean = [m / cnt for m in mean]

                var = [0.0] * d
                for point in cluster_points:
                    for j in range(d):
                        diff = point[j] - mean[j]
                        var[j] += diff * diff
                var = [v/cnt for v in var]

                self.covars[k] = var
            else:
                mean = [0.0] * d
                for point in X:
                    for j in range(d):
                        mean[j] += point[j]
                mean = [m / n for m in mean]

                var = [0.0] * d
                for point in X:
                    for j in range(d):
                        diff = point[j] - mean[j]
                        var[j] += diff * diff
                var = [v / n for v in var]

                self.covars[k] = var

            for j in range(d):
                if self.covars[k][j] < self.reg_covar:
                    self.covars[k][j] = self.reg_covar

        total = sum(self.weights)
        for k in range(self.n_gaussian):
            self.weights[k] /= total

    def _log_gauss(self, X):
        n, d = X.shape
        log_pdfs = np.empty((n, self.n_gaussian))

        for k in range(self.n_gaussian):
            var  = np.maximum(self.covars[k], self.reg_covar)
            diff = X - self.means[k]
            
            log_pdfs[:, k] = -0.5 * (d*np.log(2.*np.pi) + np.sum(np.log(var)) + np.sum(diff ** 2 / var, axis=1))
        return log_pdfs

    def _e_step(self, X):
        log_w = np.log(np.maximum(self.weights, 1e-12))
        log_pdf = self._log_gauss(X)
        log_r = log_w[None, :] + log_pdf
        log_Z = log_of_sum_exp(log_r, axis=1, keepdims=True)
        log_r -= log_Z
        total_ll = float(log_Z.sum())
        return log_r, total_ll
    
    def _m_step(self, X, log_r):
        r = np.exp(log_r)
        n_k = r.sum(axis = 0)
        self.weights = n_k/n_k.sum()

        self.means = (r.T@X)/n_k[:, None]
        for k in range(self.n_gaussian):
            diff = X - self.means[k]
            self.covars[k] = (np.sum(r[:, k, None] * diff ** 2, axis=0) / n_k[k])
            self.covars[k] = np.maximum(self.covars[k], self.reg_covar)
    
    def _fit_iter(self, X):
        self._em_start(X)
        prev_ll = -np.inf

        for iter in range(self.max_iter):
            log_r, ll = self._e_step(X)
            self._m_step(X, log_r)
            delta = ll - prev_ll
            if iter > 0 and abs(delta) < self.tol:
                break

            prev_ll = ll

        return prev_ll
    
    def fit(self, X):
        best_ll    = -np.inf
        best_state = None

        for trials in range(self.n_restarts):
            ll = self._fit_iter(X)
            if ll > best_ll:
                best_ll = ll
                best_state = {
                    "weights": self.weights.copy(),
                    "means":   self.means.copy(),
                    "covars":  self.covars.copy(),
                }
    
        self.weights = best_state["weights"]
        self.means   = best_state["means"]
        self.covars  = best_state["covars"]

    def score(self, X):
        log_pdf = self._log_gauss(X)
        log_w = np.log(np.maximum(self.weights, 1e-12))
        log_r = log_w[None, :] + log_pdf
        log_Z = log_of_sum_exp(log_r, axis = 1)
        return float(log_Z.mean())
        