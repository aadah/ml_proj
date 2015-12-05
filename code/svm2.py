import numpy as np
from sklearn.svm import SVC

import util


class MultiSVM:
    def __init__(self, K, C, kernel, kernel_param):
        self.K = K
        self.svms = [SVM(C, kernel, kernel_param) for _ in xrange(K)]


    def train(self, X, Y):
        for k in xrange(self.K):
            Y_k = Y[:,k:k+1]
            self.svms[k].train(X, Y_k)
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
            self.model = SVC(C=C, kernel=kernel, class_weight='balanced')
        elif kernel == 'poly':
            # kernel param is degree of polynomial
            self.model = SVC(C=C, kernel=kernel, degree=kernel_param, coef0=1.0, class_weight='balanced')
        elif kernel == 'rbf':
            # kernel param is gamma param of rbf
            self.model = SVC(C=C, kernel=kernel, gamma=kernel_param, class_weight='balanced')


    def train(self, X, Y, alpha_thresh=1e-6):
        Y_reshaped = Y.reshape((Y.shape[0],))
        self.model.fit(X, Y_reshaped)


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
