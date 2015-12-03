import numpy as np

import vec
import config


class DataManager:
    def __init__(self, train_filename, test_filename, meta_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename

        with open(meta_filename) as f:
            self.D = int(f.readline().strip().split()[1])
            self.K = int(f.readline().strip().split()[1])

        dossier = vec.Dossier()
        self.topics_index = dossier.topics_index
        self.terms_index = dossier.terms_index


    def load_data(self, dset='train'):
        if dset == 'train':
            filename = self.train_filename
        elif dset == 'test':
            filename = self.test_filename

        data = np.load(filename)
        X, Y = data[:,:self.D], data[:,-self.K:]
        
        return X, Y

    
    def slice_Y(self, Y, topics):
        indices = sorted([self.topics_index[topic] for topic in topics])
        Y_slice = Y[:,indices]

        return Y_slice


    def order_topics(self, topics):
        l = sorted([(self.topics_index[topic], topic) for topic in topics], key=lambda x: x[0])
        ordered = [x[-1] for x in l]

        return ordered


def create_data_manager():
    return DataManager(config.REUTERS_TRAIN, config.REUTERS_TEST, config.REUTERS_META)



