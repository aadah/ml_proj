import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from heapq import nlargest

import data

topics = [
    'earn',
    'acq',
    'money-fx',
    'grain',
    'crude',
    'trade',
    'interest',
    'ship',
    'wheat',
    'corn'
]

class Visualizer:
    
    def __init__(self, XY=None):
        self.dm = data.create_data_manager()
        if XY == None:
            self.X, self.Y = self.dm.load_data('train')
        else:
            self.X, self.Y = XY

    def PCA_transform(self, n_components=2, overwrite=False, topic='', balance='p'):
        fname = 'PCA_results/%s-X_pca_%d_components_%s.npy' %(balance, n_components, topic)
        X = self.X
        Y = self.Y
        if os.path.isfile(fname) and not overwrite:
            print 'Loading %s . . .' % fname
            self.X_pca = np.load(fname)
            N, D = self.X_pca.shape
            if N < self.X.shape[0]:
                # This means some balancing happened
                self.Y_pca = np.empty((N, 1))
                self.Y_pca[0:N/2] = np.ones((N/2, 1))
                self.Y_pca[N/2:N] = -np.ones((N/2, 1))
            else:
                self.Y_pca = self.Y
        else:
            if len(topic) > 0:
                bname = 'balanced/%s-balanced_X_%s.npy' %(balance, topic)
                if os.path.isfile(bname):
                    print 'Loading balanced X from %s . . .' % bname
                    X = np.load(bname)
                    N, _ = X.shape
                    Y = np.empty((N, 1))
                    Y[0:N/2] = np.ones((N/2, 1))
                    Y[N/2:N] = -np.ones((N/2, 1))
                else:
                    print 'Balancing for topic %s . . .' % topic
                    y = self.dm.slice_Y(self.Y, [topic])
                    X, Y = self._balance(self.X, y, 1000, balance=balance)
                    np.save(bname, X)
            print 'Running PCA . . .'
            pca = PCA(n_components=n_components)            
            self.X_pca = pca.fit_transform(X)
            self.Y_pca = Y
            np.save(fname, self.X_pca)

    def plot_X(self, topic='', n_components=2, balance='p'):
        self.PCA_transform(n_components=n_components, topic=topic, balance=balance)

        fig = plt.figure()
        if self.Y_pca.shape[1] > 1:
            if len(topic) == 0:
                print 'No topic provided!'
                return
            y = self.dm.slice_Y(self.Y_pca, [topic])
            plt.title('PCA - %s' % topic)
        else:
            y = self.Y_pca
            plt.title('PCA')
        

        y = y.reshape((y.shape[0],)) # in order for boolean mask to work

        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            for c, i, target_name in zip("rg", [-1,1], ['pos','neg']):
                ax.scatter(self.X_pca[y == i, 0], self.X_pca[y == i, 1], zs=self.X_pca[y == i, 2], c=c, label=target_name)
        else:
            for c, i, target_name in zip("rg", [-1,1], ['pos','neg']):
                plt.scatter(self.X_pca[y==i, 0], self.X_pca[y==i, 1], c=c, label=target_name)
        plt.legend()
        plt.show()

        
    def plotBoundary(self, X, Y, scoreFN, values, title=""):
        # Plot the decision boundary. For that, we will asign a score to
        # each point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = max((x_max-x_min)/200., (y_max-y_min)/200.)
        xx, yy = meshgrid(arange(x_min, x_max, h),
                          arange(y_min, y_max, h))
        zz = array([scoreFn(x) for x in c_[xx.ravel(), yy.ravel()]])
        zz = zz.reshape(xx.shape)
        pl.figure()
        CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
        pl.clabel(CS, fontsize=9, inline=1)
        # Plot the training points
        pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
        pl.title(title)
        pl.axis('tight')

    def test(self):
        iris = datasets.load_iris()
        
        X = iris.data
        y = iris.target
        target_names = iris.target_names
        
        pca = PCA(n_components=3)
        X_r = pca.fit(X).transform(X)
        
        lda = LinearDiscriminantAnalysis(n_components=3)
        X_r2 = lda.fit(X, y).transform(X)
        
        # Percentage of variance explained for each components
        print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
            ax.scatter(X_r[y == i, 0], X_r[y == i, 1], zs=X[y == i, 2], c=c, label=target_name)
        plt.legend()
        plt.title('PCA of IRIS dataset')
            
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
            ax.scatter(X_r2[y == i, 0], X_r2[y == i, 1], zs=X[y == i, 2], c=c, label=target_name)
        plt.legend()
        plt.title('LDA of IRIS dataset')
            
        plt.show()

    def _balance(self, X, Y, max_per_class, balance='p'):
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
        
        if balance == 'p':
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
        #print centroid
        distances = [(int(norm(x - centroid)*1000), x) for x in X]
        furthest = nlargest(k, distances, key=lambda e:e[0])
        new_X = np.array([e[1] for e in furthest])
        return new_X

if __name__=="__main__":
    vis = Visualizer()
    #vis.plot_X('earn', n_components=3)
    for topic in topics:
        vis.plot_X(topic, n_components=3, balance='k')
        vis.plot_X(topic, n_components=2, balance='k')
