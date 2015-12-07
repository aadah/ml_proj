import numpy as np
from pprint import pprint

import svm # our implementation
import svm2 # sklearn implementation
import knn
import data
import evaluate


np.random.seed(100)


# SVM hyperparameters
svm_Cs = [1, 10, 100]

# for our implementation
svm_kernels = [
    #('linear', svm.linear_kernel),
    ('poly(1)', svm.make_polynomial_kernel(1)),
    ('poly(2)', svm.make_polynomial_kernel(2)),
    ('poly(3)', svm.make_polynomial_kernel(3)),
    ('poly(4)', svm.make_polynomial_kernel(4)),
    ('poly(5)', svm.make_polynomial_kernel(5)),
    ('rbf(0.6)', svm.make_gaussian_kernel(0.6)),
    ('rbf(0.8)', svm.make_gaussian_kernel(0.8)),
    ('rbf(1.0)', svm.make_gaussian_kernel(1.0)),
    ('rbf(1.2)', svm.make_gaussian_kernel(1.2))
]

# for sklearn implementation
svm2_kernels = [
    #('linear', None),
    ('poly', 1),
    ('poly', 2),
    ('poly', 3),
    ('poly', 4),
    ('poly', 5),
    ('rbf', 0.6),
    ('rbf', 0.8),
    ('rbf', 1.0),
    ('rbf', 1.2)
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
    # test run using degree-3 polynomial kernel with C=1
    # for the ten main topics in reuters
    dm = data.create_data_manager()

    print 'loading train data . . .'
    X_train, Y_train = dm.load_data('train')
    Y_train_slice = dm.slice_Y(Y_train, topics)

    print 'training svm . . .'
    K = len(topics)
    learner = svm2.MultiSVM(K, 1.0, 'poly', 3)
    learner.train(X_train, Y_train_slice)

    print 'loading test data . . .'
    X_test, Y_test = dm.load_data('test')
    Y_gold = dm.slice_Y(Y_test, topics)

    print 'predicting train . . .'
    Y_pred = learner.batch_predict_classes(X_train)
    
    print 'evaluating . . .'
    precision, recall = evaluate.precision_recall(Y_pred, Y_train_slice)
    f1 = evaluate.f_score(precision, recall)

    print 'Precision: %.3f' % precision
    print 'Recall: %.3f' % recall
    print 'F1: %.3f' % f1

    print 'predicting test . . .'
    Y_pred = learner.batch_predict_classes(X_test)
    
    print 'evaluating . . .'
    precision, recall = evaluate.precision_recall(Y_pred, Y_gold)
    f1 = evaluate.f_score(precision, recall)

    print 'Precision: %.3f' % precision
    print 'Recall: %.3f' % recall
    print 'F1: %.3f' % f1


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


def svm_experiment(C):
    K = len(topics)
    dm = data.create_data_manager()
    ordered_topics = dm.order_topics(topics)

    print 'loading train data . . .'
    X_train, Y_train = dm.load_data('train')
    Y_train_slice = dm.slice_Y(Y_train, ordered_topics)

    print 'loading test data . . .'
    X_test, Y_test = dm.load_data('test')
    Y_gold = dm.slice_Y(Y_test, ordered_topics)

    final_results = {}

    #print (Y_train_slice.sum(axis=0) + Y_train_slice.shape[0]).tolist()
    #raise Exception

    print 'interating over models . . .'
    for model, kernel in svm_kernels:
        print 'now using model %s . . .' % model
        learner = svm.MultiSVM(K, C, kernel)
        learner.train(X_train, Y_train_slice,
                      balance=True,
                      max_per_class=3000)
        Y_pred = learner.batch_predict_classes(X_test)

        results = evaluate.per_topic_results(Y_pred, Y_gold)
        results_dict = {topic: result for (topic, result) in zip(ordered_topics, results)}

        pprint(results_dict)
        final_results[model] = results_dict

    print 'saving final results . . .'
    with open('results/svm_final_results_C_%.1f_.txt' % C, 'w') as f:
        pprint(final_results, stream=f)


def svm2_experiment(C):
    K = len(topics)
    dm = data.create_data_manager()
    ordered_topics = dm.order_topics(topics)

    print 'loading train data . . .'
    X_train, Y_train = dm.load_data('train')
    Y_train_slice = dm.slice_Y(Y_train, ordered_topics)

    print 'loading test data . . .'
    X_test, Y_test = dm.load_data('test')
    Y_gold = dm.slice_Y(Y_test, ordered_topics)

    final_results = {}

    print 'interating over models . . .'
    for kernel, kernel_param in svm2_kernels:
        model = '%s(%s)' % (kernel, str(kernel_param))

        print 'now using model %s . . .' % model
        learner = svm2.MultiSVM(K, C, kernel, kernel_param)
        learner.train(X_train, Y_train_slice,
                      balance=True,
                      max_per_class=3000)
        Y_pred = learner.batch_predict_classes(X_test)

        results = evaluate.per_topic_results(Y_pred, Y_gold)
        results_dict = {topic: result for (topic, result) in zip(ordered_topics, results)}

        pprint(results_dict)
        final_results[model] = results_dict

    print 'saving final results . . .'
    with open('results/svm2_final_results_C_%.1f_partial_new_balance.txt' % C, 'w') as f:
        pprint(final_results, stream=f)


def knn_experiement():
    pass


if __name__ == '__main__':
    #test()
    #svm2_experiment(1000000000.0) # all data
    #svm_experiment(1000000000.0) # part of data
    svm2_experiment(1000000000.0) # part of data
