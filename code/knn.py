import os.path
import numpy as np
import scipy.spatial as sps
import data

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
        #return np.apply_along_axis(self.predict_classes, 1, X)
        return np.hstack([self.knns[k].batch_predict_class(X) for k in xrange(self.K)])

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
        pass
        '''
        results = [(self._distance(x, self.X[i]), self.Y[i][0]) for i in xrange(self.X.shape[0])]
        results.sort(key=lambda r: r[0])
        top = results[:self.k]

        count_one = len(filter(lambda r: r[1] == 1.0, top))
        count_two = len(top) - count_one
        
        return 1.0 if count_one > count_two else -1.0
        '''

    def batch_predict_class(self, X):
        #return np.apply_along_axis(self.predict_class, 1, X).reshape((len(X), 1))
        N, _ = X.shape
        top_indices = self.top_indices()
        k_indices = top_indices[:,:self.k]
        Y_pred = np.array([np.sign(sum([self.Y[i][0] for i in k_indices[j]])) for j in xrange(N)]).reshape((N,1))
        return Y_pred
        

    def classification_error(self, X, Y):
        N, _ = X.shape

        Y_pred = self.batch_predict_class(X)
        diff = Y_pred == Y
        
        incorrect = len(filter(lambda x: not x, diff))

        return float(incorrect) / N

    def _distance(self, u, v):
        return sps.distance.euclidean(u, v)

    def distance_matrix(self):
        print 'creating distance_matrix . . .'
        dm = data.create_data_manager()
        X_train, _ =  dm.load_data('train')
        X_test, _ = dm.load_data('test')
        
        distances = np.empty((X_test.shape[0], X_train.shape[0]))
        for row in xrange(X_test.shape[0]):
            print 'row', row
            x = X_test[row]
            distances[row] = np.array([self._distance(x, X_train[col]) for col in xrange(X_train.shape[0])])
        np.save('distance_matrix.npy', distances)
        return distances
    
    def top_indices(self):
        if os.path.isfile('top_indices.npy'):
            print 'loading top_indices . . .'
            top_indices = np.load('top_indices.npy')
            return top_indices
        print 'creating top_indices . . .'
        distance_matrix = None
        if not os.path.isfile('distance_matrix.npy'):            
            distance_matrix = self.distance_matrix()
        else:
            print 'loading distance_matrix . . .'
            distance_matrix = np.load('distance_matrix.npy')
        N_test, N_train = distance_matrix.shape
        top_indices = np.empty((N_test, N_train))
        for j in xrange(N_test):
            print 'row', j
            row = distance_matrix[j]
            tagged_row = [(row[i], i) for i in xrange(N_train)]
            tagged_row.sort(key=lambda d: d[0]) # sort by distance
            top_indices[j] = np.array([d[1] for d in tagged_row])
        np.save('top_indices.npy', top_indices)
        return top_indices
        
if __name__=='__main__':
    knn_model = kNN(0)
    knn_model.top_indices()
