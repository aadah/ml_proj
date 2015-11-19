import numpy as np
import scipy.spatial as sps


class MultikNN:
    def __init__(self, K, k):
        self.K = K
        self.knns = [kNN(k) for _ in xrange(K)]


    def train(self, X, Y):
        for k in xrange(self.K):
            Y_k = Y[:,k:k+1]
            self.knns[k].train(X, Y_k)


    def predict_classes(self, x):
        return np.array([self.knns[k].predict_class(x) for k in xrange(self.K)])


    def batch_predict_classes(self, X):
        return np.apply_along_axis(self.predict_classes, 1, X)


    def classification_errors(self, X, Y):
        N, _ = X.shape
        errors = np.empty(self.K)

        for k in xrange(self.K):
            Y_k = Y[:,k:k+1]
            Y_pred = self.knns[k].batch_predict_class(X)

            diff = Y_pred == Y_k
            incorrect = len(filter(lambda x: not x, diff))

            errors[k] = float(incorrect) / N

        return errors


class kNN:
    def __init__(self, k):
        self.k = k
        
        #params
        self.X = None
        self.Y = None


    def train(self, X, Y):
        self.X = X
        self.Y = Y


    def predict_class(self, x):
        results = [(self._distance(x, self.X[i]), self.Y[i][0]) for i in xrange(self.X.shape[0])]
        results.sort(key=lambda r: r[0])
        top = results[:self.k]

        count_one = len(filter(lambda r: r[1] == 1.0, top))
        count_two = len(top) - count_one

        return 1.0 if count_one > count_two else -1.0


    def batch_predict_class(self, X):
        return np.apply_along_axis(self.predict_class, 1, X).reshape((len(X), 1))


    def classification_error(self, X, Y):
        N, _ = X.shape

        Y_pred = self.batch_predict_class(X)
        diff = Y_pred == Y
        
        incorrect = len(filter(lambda x: not x, diff))

        return float(incorrect) / N


    def _distance(self, u, v):
        return sps.distance.cosine(u, v)
