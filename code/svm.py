import numpy as np
from cvxopt import matrix, solvers


solvers.options['show_progress'] = False


class MultiSVM:
    def __init__(self, K, kernel, C):
        self.K = K
        self.svms = [SVM(kernel, C) for _ in xrange(K)]


    def train(self, X, Y):
        for k in xrange(self.K):
            print '%d...' % k,
            Y_k = Y[:,k:k+1]
            self.svms[k].train(X, Y_k)


    def predict_margins(self, x):
        return np.array([self.svms[k].predict_margin(x) for k in xrange(self.K)])


    def batch_predict_margins(self, X):
        return np.apply_along_axis(self.predict_margins, 1, X)


    def predict_classes(self, x):
        return np.array([self.svms[k].predict_class(x) for k in xrange(self.K)])


    def batch_predict_classes(self, X):
        return np.apply_along_axis(self.predict_classes, 1, X)


    def classification_errors(self, X, Y):
        N, _ = X.shape
        errors = np.empty(self.K)

        for k in xrange(self.K):
            Y_k = Y[:,k:k+1]
            Y_pred = self.svms[k].batch_predict_class(X)

            diff = Y_pred == Y_k
            incorrect = len(filter(lambda x: not x, diff))

            errors[k] = float(incorrect) / N

        return errors


class SVM:
    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C

        #params
        self.alphas = None
        self.support_targets = None
        self.support_vectors = None


    def train(self, X, Y, alpha_thresh=1e-6):
        self.alphas = []
        self.support_targets = []
        self.support_vectors = []

        X_pad = np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)
        
        args = self._make_svm_params(X_pad, Y)
        sol = solvers.qp(*args)
        alphas = np.array(sol['x'])
        triples = [triple for triple in zip(alphas, Y, X_pad) if triple[0][0] >= alpha_thresh]

        for (alpha, y, x) in triples:
            self.alphas.append(alpha[0])
            self.support_targets.append(y[0])
            self.support_vectors.append(x)

    
    def predict_margin(self, x):
        x_pad = np.pad(x, (1,0), 'constant', constant_values=1)
        triples = zip(self.alphas, self.support_targets, self.support_vectors)
        margin = sum([alpha * y * self.kernel(x_pad, x) for (alpha, y, x) in triples])

        return margin


    def batch_predict_margin(self, X):
        return np.apply_along_axis(self.predict_margin, 1, X).reshape((len(X), 1))

        
    def predict_class(self, x):
        margin = self.predict_margin(x)
    
        return 1.0 if margin > 0.0 else -1.0


    def batch_predict_class(self, X):
        return np.apply_along_axis(self.predict_class, 1, X).reshape((len(X), 1))


    def classification_error(self, X, Y):
        N, _ = X.shape

        Y_pred = self.batch_predict_class(X)
        diff = Y_pred == Y
        
        incorrect = len(filter(lambda x: not x, diff))

        return float(incorrect) / N


    def _make_P(self, X, Y):
        N = X.shape[0]
        P = np.dot(Y, Y.T) #np.zeros((N,N))
        
        
        for i in xrange(N):
            for j in xrange(N):
                #P[i,j] = self.kernel(X[i],X[j]) * Y[i] * Y[j]
                P[i,j] *= self.kernel(X[i],X[j])

        return P


    def _make_svm_params(self, X, Y):
        n = X.shape[0]

        P = self._make_P(X, Y)
        q = -np.ones(n)
        G = np.vstack((-np.identity(n), np.identity(n)))
        h = np.concatenate((np.zeros(n), self.C * np.ones(n)))
        A = np.array(Y.T)
        b = np.zeros(1)

        return (matrix(P),
                matrix(q),
                matrix(G),
                matrix(h),
                matrix(A),
                matrix(b))


# kernels

def linear_kernel(u, v):
    return np.dot(u, v)


def make_gaussian_kernel(bw):

    def gaussian_kernel(u, v):
        diff = u - v
        return np.exp(- np.dot(diff, diff) / (2 * pow(bw, 2)))

    return gaussian_kernel


def make_polynomial_kernel(d):
    
    def polynomial_kernel(u, v):
        return np.power(np.dot(u, v) + 1, d)

    return polynomial_kernel
