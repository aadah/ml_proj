import os.path
import numpy as np
from numpy.linalg import norm
from cvxopt import matrix, solvers
from heapq import nlargest

import util

solvers.options['show_progress'] = False


class MultiSVM:
    def __init__(self, K, C, kernel):
        self.K = K
        self.svms = [SVM(C, kernel) for _ in xrange(K)]


    def train(self, X, Y, 
              ordered_topics, 
              balance='k', 
              max_per_class=100):
        topic = ''
        for k in xrange(self.K):
            topic = ordered_topics[k]
            Y_k = Y[:,k:k+1]
            self.svms[k].train(X, Y_k, 
                               balance=balance, 
                               max_per_class=max_per_class,
                               topic=topic)
            util.my_print('%d . . .' % (k+1), same_line=True)


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
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel

        #params
        self.alphas = None
        self.support_targets = None
        self.support_vectors = None


    def train(self, X, Y, alpha_thresh=1e-6,
              balance='k',
              max_per_class=100,
              topic=''):

        self.alphas = []
        self.support_targets = []
        self.support_vectors = []

        if len(balance) > 0:
            bname = 'balanced/%s-balanced_X_%s.npy' %(balance, topic)
            if os.path.isfile(bname):
                print 'Loading balanced X from %s . . .' % bname
                X = np.load(bname)
                N,_ = X.shape
                Y = np.empty((N, 1))
                Y[0:N/2] = np.ones((N/2, 1))
                Y[N/2:N] = -np.ones((N/2, 1))
            else:
                X, Y = self._balance(X, Y, max_per_class, balance=balance)
                np.save(bname, X)

            print 'X dim:', X.shape

        X = np.pad(X, ((0,0),(1,0)), 'constant', constant_values=1)
        
        args = self._make_svm_params(X, Y)
        sol = solvers.qp(*args)
        alphas = np.array(sol['x'])
        triples = [triple for triple in zip(alphas, Y, X) if triple[0][0] >= alpha_thresh]

        for (alpha, y, x) in triples:
            self.alphas.append(alpha[0])
            self.support_targets.append(y[0])
            self.support_vectors.append(x)

    def _balance(self, X, Y, max_per_class):
        # X is the original matrix of training points
        # Y is an Nx1 array, where y^(i) indicates if x^(i) is
        #     a member of the topic represented by this column
        #     in the full Nx90 matrix
        N, D = X.shape
        tally = Y.sum()

        if tally == 0: # already balanced
            num_pos = min(N / 2, max_per_class)
            num_neg = num_pos

        elif tally > 0: # more positive than negative
            num_neg = min((N-tally) / 2, max_per_class)
            num_pos = num_neg
                
        elif tally < 0: # more negative than positive
            num_pos = min((N-abs(tally)) / 2, max_per_class)
            num_neg = num_pos

        if balance == 'p':
            new_X = np.empty((num_pos+num_neg, D))
            new_Y = np.empty((num_pos+num_neg, 1))
            
            pos_data = X[Y[:, 0] == 1,:]
            neg_data = X[Y[:, 0] == -1,:]
        
            new_pos = self._get_k_furthest(pos_data, int(num_pos))
            new_neg = self._get_k_furthest(neg_data, int(num_neg))
        
            new_X[0:num_pos] = new_pos
            new_Y[0:num_pos] = np.ones((num_pos,1))
            new_X[num_pos:num_pos+num_neg] = new_neg
            new_Y[num_pos:num_pos+num_neg] = -np.ones((num_neg,1))
        
        elif balance == 'k':
            # original rebalancer
            i = 0
            for n in xrange(N):
                if Y[n][0] == 1 and num_pos > 0:
                    new_X[i] = X[n]
                    new_Y[i] = Y[n]
                    num_pos -= 1
                    i += 1
                elif Y[n][0] == -1 and num_neg > 0:
                    new_X[i] = X[n]
                    new_Y[i] = Y[n]
                    num_neg -= 1
                    i += 1

        return new_X, new_Y

    def _get_k_furthest(self, X, k):
        N, D = X.shape
        if N == k:
            return X
        distances = []
        new_X = np.empty((k, D))
        # a mapping from distance to the points with that distance from centroid
        centroid = np.mean(X, axis=0)
        distances = [(int(norm(x - centroid)*1000), x) for x in X]
        furthest = nlargest(k, distances, key=lambda e:e[0])
        new_X = np.array([e[1] for e in furthest])
        return new_X

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
        #P = np.zeros((N,N))
        P = np.dot(Y, Y.T)
        
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


def make_gaussian_kernel(gamma):

    def gaussian_kernel(u, v):
        diff = u - v
        return np.exp(- gamma * np.dot(diff, diff))

    return gaussian_kernel


def make_polynomial_kernel(d):
    
    def polynomial_kernel(u, v):
        return np.power(np.dot(u, v) + 1, d)

    return polynomial_kernel
