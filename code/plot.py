import pylab as pl
import random
import experiments


def load_results_as_dict(filename):
    with open(filename) as f:
        string = f.read()

    return eval(string)


def plot_precision_recall_per_topic(results, model):
    random.seed(10000)

    colors = ['b','g','r','y','c','m']
    symbols = ['o','^','s','p','*','H','D']
    markers = [color+symbol for color in colors for symbol in symbols]
    random.shuffle(markers)

    topics = experiments.topics
    x_points = [results[model][topic][0] for topic in topics] # precision
    y_points = [results[model][topic][1] for topic in topics] # recall

    x_max = max(x_points)
    x_min = min(x_points)
    y_max = max(y_points)
    y_min = min(y_points)

    x_buf = (x_max - x_min)*0.05
    y_buf = (y_max - y_min)*0.05

    pl.figure()
    pl.title("Precision vs. Recall for model %s" % model)
    pl.xlabel('Precision')
    pl.ylabel('Recall')
    pl.xlim([x_min-x_buf, x_max+x_buf])
    pl.ylim([y_min-y_buf, y_max+y_buf])

    for (topic, marker, x, y) in zip(topics, markers, x_points, y_points):
        pl.plot([x],[y],marker,label=topic,markersize=15)

    pl.legend(loc='best')
    pl.show()


def plot_precision_recall_per_model(results, topic):
    random.seed(10000)

    colors = ['b','g','r','y','c','m']
    symbols = ['o','^','s','p','*','H','D']
    markers = [color+symbol for color in colors for symbol in symbols]
    random.shuffle(markers)

    models = results.keys()
    x_points = [results[model][topic][0] for model in models] # precision
    y_points = [results[model][topic][1] for model in models] # recall

    x_max = max(x_points)
    x_min = min(x_points)
    y_max = max(y_points)
    y_min = min(y_points)

    x_buf = (x_max - x_min)*0.05
    y_buf = (y_max - y_min)*0.05

    pl.figure()
    pl.title("Precision vs. Recall for topic '%s'" % topic)
    pl.xlabel('Precision')
    pl.ylabel('Recall')
    pl.xlim([x_min-x_buf, x_max+x_buf])
    pl.ylim([y_min-y_buf, y_max+y_buf])

    for (model, marker, x, y) in zip(models, markers, x_points, y_points):
        pl.plot([x],[y],marker,label=model,markersize=15)

    pl.legend(loc='best')
    pl.show()
