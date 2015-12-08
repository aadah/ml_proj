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

    print true_pos, true_neg, false_pos, false_neg
    
    if true_pos + false_pos > 0:
        precision = true_pos / (true_pos + false_pos)
    else:
        precision = 0.0

    recall = true_pos / (true_pos + false_neg)

    return (precision, recall)


def per_topic_results(Y_pred, Y_gold, beta=1):
    all_results = []
    _, K = Y_gold.shape

    for k in xrange(K):
        p, r = precision_recall(Y_pred[:,k:k+1], Y_gold[:,k:k+1])
        f = f_score(p, r, beta=beta)
        all_results.append((p,r,f))

    return all_results


def f_score(precision, recall, beta=1):
    beta_squared = pow(beta, 2)
    coef = 1 + beta_squared
    score = coef * (precision*recall) / (beta_squared*precision + recall)

    return score
