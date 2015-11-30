import numpy as np


def precision_recall(Y_pred, Y_gold):
    true_pos = 0.0
    true_neg = 0.0
    false_pos = 0.0
    false_neg = 0.0
    
    assert Y_pred.shape == Y_gold.shape
    
    N, K = Y_gold.shape

    for n in xrange(N):
        for k in xrange(K):
            y_pred = Y_pred[n,k]
            y_gold = Y_gold[n,k]

            if y_gold == -1:
                if y_pred == -1:
                    true_neg += 1
                elif y_pred == 1:
                    false_pos += 1
                else:
                    raise Exception('Prediction is not in {-1,1}')
            elif y_gold == 1:
                if y_pred == -1:
                    false_neg += 1
                elif y_pred == 1:
                    true_pos += 1
                else:
                    raise Exception('Prediction is not in {-1,1}')
            else:
                raise Exception('Gold label is not in {-1,1}')

    np.save('y_pred.npy', Y_pred)
    np.save('y_gold.npy', Y_gold)

    print true_pos, true_neg, false_pos, false_neg
    
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    return (precision, recall)


def split_precision_recall(Y_pred, Y_gold):
    pass


def f_score(precision, recall, beta=1):
    beta_squared = pow(beta, 2)
    coef = 1 + beta_squared
    score = coef * (precision*recall) / (beta_squared*precision + recall)

    return score
