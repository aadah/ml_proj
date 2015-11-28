import svm
import knn
import data


# SVM hyperparameters
svm_Cs = [1000000000, 0.01, 0.1, 1, 10, 100]
svm_kernels = [
    ('linear', svm.linear_kernel),
    ('poly1', svm.make_polynomial_kernel(1)),
    ('poly2', svm.make_polynomial_kernel(2)),
    ('poly3', svm.make_polynomial_kernel(3)),
    ('poly4', svm.make_polynomial_kernel(4)),
    ('poly5', svm.make_polynomial_kernel(5)),
    ('rbf0.6', svm.make_gaussian_kernel(0.6)),
    ('rbf0.8', svm.make_gaussian_kernel(0.8)),
    ('rbf1.0', svm.make_gaussian_kernel(1.0)),
    ('rbf1.2', svm.make_gaussian_kernel(1.2)),
]

# kNN hyperparameters
knn_ks = [1, 11, 21, 31, 41, 51]

# main topics
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


def test():
    print 'loading data . . .'
    dm = data.create_data_manager()
    X_train, Y_train = dm.load_data('train')
    Y_train_slice = dm.slice_Y(Y_train, topics)

    print 'training svm . . .'
    K = len(topics)
    learner = svm.MultiSVM(K, svm.make_polynomial_kernel(3), 1.0)
    learner.train(X_train, Y_train_slice)


def test2():
    print 'loading data . . .'
    dm = data.create_data_manager()
    X_train, Y_train = dm.load_data('train')
    Y_train_slice = dm.slice_Y(Y_train, topics)

    print 'training knn . . .'
    K = len(topics)
    learner = knn.MultikNN(K, 30)
    learner.train(X_train, Y_train_slice)

    print 'testing knn . . .'
    X_test, Y_test = dm.load_data('test')
    Y_test_slice = dm.slice_Y(Y_test, topics)
    errors = learner.classification_errors(X_test, Y_test_slice)

    return errors


def svm_experiment():
    pass


def knn_experiement():
    pass
