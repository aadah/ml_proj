import numpy as np
from numpy.linalg import norm
from sklearn.svm import SVC
from heapq import nlargest

import util


class MultiSVM:
    def __init__(self, K, C, kernel, kernel_param):
        self.K = K
        self.svms = [SVM(C, kernel, kernel_param) for _ in xrange(K)]


    def train(self, X, Y,
              balance=False,
              max_per_class=100):
        for k in xrange(self.K):
            Y_k = Y[:,k:k+1]
            self.svms[k].train(X, Y_k,
                               balance=balance,
                               max_per_class=max_per_class)
            util.my_print('%d . . .' % (k+1), same_line=True)


    def batch_predict_classes(self, X):
        Y_pred = np.hstack([self.svms[k].batch_predict_class(X) for k in xrange(self.K)])
        
        return Y_pred


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
    def __init__(self, C, kernel, kernel_param):
        self.C = C
        self.kernel = kernel
        self.kernel_param = kernel_param

        if kernel == 'linear':
            self.model = SVC(C=C, kernel=kernel)
        elif kernel == 'poly':
            # kernel param is degree of polynomial
            self.model = SVC(C=C, kernel=kernel, degree=kernel_param, coef0=1.0)
        elif kernel == 'rbf':
            # kernel param is gamma param of rbf
            self.model = SVC(C=C, kernel=kernel, gamma=kernel_param)


    def train(self, X, Y, alpha_thresh=1e-6,
              balance=False,
              max_per_class=100):
        if balance:
            X, Y = self._balance(X, Y, max_per_class)
            print 'X dim:', X.shape

        Y_reshaped = Y.reshape((Y.shape[0],))
        self.model.fit(X, Y_reshaped)


    def _balance(self, X, Y, max_per_class):
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
        
        '''        
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
        '''
        return new_X, new_Y

    def _get_k_furthest(self, X, k):
        N, D = X.shape
        if N == k:
            return X
        distances = []
        new_X = np.empty((k, D))
        # a mapping from distance to the points with that distance from centroid
        centroid = np.mean(X, axis=0)
        #print centroid
        distances = [(int(norm(x - centroid)*1000), x) for x in X]
        furthest = nlargest(k, distances, key=lambda e:e[0])
        new_X = np.array([e[1] for e in furthest])
        return new_X

    def batch_predict_class(self, X):
        Y = self.model.predict(X)
        Y_reshaped = Y.reshape((Y.shape[0],1))

        return Y_reshaped


    def classification_error(self, X, Y):
        N, _ = X.shape

        Y_pred = self.batch_predict_class(X)
        diff = Y_pred == Y
        
        incorrect = len(filter(lambda x: not x, diff))

        return float(incorrect) / N
