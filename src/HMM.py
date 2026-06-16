import numpy as np
from utils import log_of_sum_exp, VectorQuantizer

class HMM:
    def __init__(self, n_states = 5, n_symbols = 64, max_iter = 20, tol = 1.):
        self.n_states  = n_states
        self.n_symbols = n_symbols
        self.max_iter  = max_iter
        self.tol = tol
        self.log_pi = None
        self.log_A = None
        self.log_B = None

    def _bm_start(self):
        S = self.n_states
        M = self.n_symbols

        pi = np.zeros(S)
        pi[0] = 1.0
        self.log_pi = np.log(np.maximum(pi, 1e-12))

        A = np.zeros((S, S))
        for i in range(S-1):
            A[i, i] = 0.5
            A[i, i+1] = 0.5
        A[S-1, S-1] = 1.0
        self.log_A = np.log(np.maximum(A, 1e-12))

        B = np.ones((S, M)) + np.random.rand(S, M) * 0.1
        for i in range(S):
            B_sum = np.sum(B[i])
            for j in range(M):
                B[i, j] /= B_sum
        self.log_B = np.log(np.maximum(B, 1e-12))

        self.log_pi = np.array(self.log_pi)
        self.log_A = np.array(self.log_A)
        self.log_B = np.array(self.log_B)

    def _log_emission(self, X):
        T = len(X)
        log_B = np.empty((T, self.n_states))

        for t in range(T):
            for j in range(self.n_states):
                log_B[t, j] = self.log_B[j, X[t]]

        return log_B

    def _forward(self, log_B):
        T, S   = log_B.shape
        log_alpha = np.empty((T, S))

        log_alpha[0] = self.log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = log_of_sum_exp(log_alpha[t-1, :, None]+ self.log_A, axis=0) + log_B[t]

        log_ll = float(log_of_sum_exp(log_alpha[-1]))
        return log_alpha, log_ll

    def _backward(self, log_B):
        T, S   = log_B.shape
        log_beta = np.zeros((T, S))

        for t in range(T - 2, -1, -1):
            log_beta[t] = log_of_sum_exp(self.log_A + log_B[t+1] + log_beta[t+1], axis = 1)

        return log_beta

    def fit(self, X_list):
        self._bm_start()
        S = self.n_states
        M = self.n_symbols
        prev_ll = -np.inf

        for iter in range(self.max_iter):
            total_ll = 0.0

            log_xi_num = np.full((S, S), -np.inf)
            log_xi_den = np.full(S, -np.inf)

            B_num = np.ones((S, M)) * 1e-6
            B_den = np.ones(S) * (M * 1e-6)

            for X in X_list:
                T = len(X)
                log_B = self._log_emission(X)
                log_alpha, seq_ll = self._forward(log_B)
                log_beta  = self._backward(log_B)
                total_ll += seq_ll

                log_gamma = log_alpha + log_beta - seq_ll
                gamma = np.exp(log_gamma)

                for j in range(S):
                    B_den[j] += np.sum(gamma[:, j])

                for t in range(T):
                    symbol = X[t]
                    for j in range(S):
                        B_num[j, symbol] += gamma[t, j]

                log_xi_seq = log_alpha[:-1, :, None] + self.log_A[None, :, :] +\
                            log_B[1:, None, :] + log_beta[1:, None, :] - seq_ll

                lx_sum = log_of_sum_exp(log_xi_seq, axis = 0)
                log_xi_num = log_of_sum_exp(np.stack([log_xi_num, lx_sum]), axis=0)

                lg_den = log_of_sum_exp(log_gamma[:-1], axis=0)
                log_xi_den = log_of_sum_exp(np.stack([log_xi_den, lg_den]), axis=0)

            for i in range(S):
                for j in range(S):
                    if j >= i:
                        self.log_A[i, j] = log_xi_num[i, j] - log_xi_den[i]
                    else:
                        self.log_A[i, j] = -np.inf
                self.log_A[i] -= log_of_sum_exp(self.log_A[i])

            for i in range(S):
                for j in range(M):
                    self.log_B[i, j] = np.log(B_num[i, j] / B_den[i])

            delta = total_ll - prev_ll
            if iter > 0 and abs(delta) < self.tol:
                break

            prev_ll = total_ll

    def score(self, X):
        log_B = self._log_emission(X)
        _, log_ll = self._forward(log_B)
        return log_ll / len(X)
