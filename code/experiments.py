import svm
import knn
#import [FILE FOR LOADING DATA]


# SVM hyperparameters
svm_Cs = [1000000000, 0.01, 0.1, 1, 10, 100]
svm_kernels = [
    ('linear', svm.linear_kernel),
    ('poly3', svm.make_polynomial_kernel(3)),
    ('poly5', svm.make_polynomial_kernel(5)),
    ('poly7', svm.make_polynomial_kernel(7)),
    ('rbf0.1', svm.make_gaussian_kernel(0.1)),
    ('rbf1', svm.make_gaussian_kernel(1)),
    ('rbf10', svm.make_gaussian_kernel(10)),
]

# kNN hyperparameters
knn_ks = [1, 11, 21, 31, 41, 51]


def svm_experiment():
    pass


def knn_experiement():
    pass
