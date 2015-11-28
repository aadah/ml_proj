#!/usr/bin/env python

from bs4 import BeautifulSoup
import nltk
import re
import os  
import config
import collections
import numpy as np
import sklearn.preprocessing as skp

STOPWORDS = nltk.corpus.stopwords.words('english')
STEMMER = nltk.stem.SnowballStemmer('english')

TOPICS_FILE = 'topics.txt'
TERMS_FILE = 'terms.txt'

class Article:
    
    def __init__(self, text, metaDict, topics):
        #self.text = text
        # body tags cannot be accessed by BeautifulSoup
        #print text
        #soup = BeautifulSoup(text.replace("BODY","CONTENT"), 'lxml')
        #self.date = soup.date
        #self.topics = soup.topics
        #print self.topics
        '''
        self.places = soup.places
        self.people = soup.people
        self.orgs = soup.orgs
        self.exchanges = soup.exchanges
        self.companies = soup.companies
        self.title = soup.title
        self.dateline = soup.dateline
        '''
        #text = soup.content
        #self.text = text
        #self.words = self.stemAndRemoveStopwords(text)
        self.topics = topics
        self.word_bag = self.getBagOfWords(text)
        #print topics
        self.id = int(metaDict['newid'])
        self.split = metaDict['lewissplit']
        #print self.id

    def getId(self):
        return self.id

    def stemAndRemoveStopwords(self, text):
        new_text = ''
        for word in text.split():
            if word not in STOPWORDS:
                new_text += STEMMER.stem(word) + ' '
        return new_text

    '''
    def stemWords(self, text):
        new_text = ''
        for word in text:
            new_text += STEMMER.stem(word) + ' '
        return new_text
    '''

    def getBagOfWords(self, text=''):
        word_count = collections.defaultdict(int)
        word_list = []
        text = re.sub('[^a-z]',' ',text.lower())
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            #word_list += [STEMMER.stem(w) for w in words if w not in STOPWORDS] # a copy of the results from this is backed up
            word_list += [STEMMER.stem(w) for w in words if STEMMER.stem(w) not in STOPWORDS]
        for word in word_list:
            word_count[word] += 1
        return word_count

    def getVector(self, terms_index, topics_index):
        num_words = len(terms_index)
        num_topics = len(topics_index)
        vector = np.zeros(num_words + num_topics)
        for word in self.word_bag:
            if word in terms_index:
                vector[terms_index[word]] = self.word_bag[word]
        for topic in self.topics:
            if topic in topics_index:
                vector[num_words + topics_index[topic]] = 1
        return vector


class Dossier:
    
    def __init__(self, dirName=None):
        self.articles = {}
        self.topics_index = {}
        self.readResource('topics')
        self.terms_index = {}
        self.readResource('terms')
        print self.topics_index

        if dirName is not None:
            self.readDir(dirName)
    
    def addArticle(self, article):
        # Takes in an Article object and adds to its dictionary of articles
        # Articles are keyed by their id, which is the "NEWID" field in each
        # article's header
        articleId = article.getId()
        if articleId in self.articles:
            print "WARNING: ID COLLISION DETECTED FOR ID: " + articleId
            altId = articleId + '_collision'
            self.articles[altId] = article
            print "ARTICLE STORED WITH ID: " + altId
        else:
            self.articles[articleId] = article

    def getArticle(self, id):
        return self.articles[id]

    def readSgm(self, fname):
        with open(fname, 'r') as f:
            inArticle = False
            inBody = False
            meta = ''
            text = ''
            topics = []
            metaDict = None
            count = 0
            for line in f:
                if inArticle:
                    if line.startswith('<TOPICS>'):
                        if metaDict and self.isModApte(metaDict):
                            topics = re.sub(r'<(/?)D>',' ',line[8:-10]).split()
                    elif '<BODY>' in line:
                        inBody = True
                    elif '</BODY>' in line:
                        inBody = False
                        text = text[text.index('<BODY>')+6:]
                    if inBody:
                        # adding after prevents adding final line, which has no text
                        text += line
                    if line.startswith("</REUTERS>"):
                        inArticle = False
                        a = Article(text, metaDict, topics)
                        self.addArticle(a)
                        count += 1
                        #print 'id', a.getId()
                        #if count > 10:
                        #    return
                elif line.startswith("<REUTERS"):
                    meta = line
                    # check if the article is used for the ModApte split
                    metaDict = self.readMeta(meta)
                    if metaDict:
                        inArticle = True
                    
    def learnTopics(self, fname):
        train_topics = set()
        test_topics = set()
        with open(fname, 'r') as f:
            for line in f:
                if line.startswith('<REUTERS'):
                    metaDict = self.readMeta(line)
                    split = ''
                    if metaDict:
                        split = metaDict['lewissplit']
                elif line.startswith('<TOPICS>'):
                    if metaDict and self.isModApte(metaDict):
                        topics = re.sub(r'<(/?)D>',' ',line[8:-10]).split()
                        if split == '':
                            continue
                        elif split == 'TRAIN':
                            train_topics.update(set(topics))
                        elif split == 'TEST':
                            test_topics.update(set(topics))
        return train_topics, test_topics
    
    def learnTerms(self, fname):
        terms = set()
        word_count = collections.defaultdict(int)
        with open(fname, 'r') as f:
            meta = ''
            text = ''
            inText = False
            for line in f:
                if line.startswith('<TEXT>'):
                    inText = True
                elif inText:
                    text += line                    
                    if line.endswith('</TEXT>\n'):
                        if (metaDict 
                            and self.isModApte(metaDict) 
                            and  metaDict['lewissplit'] == 'TRAIN'):
                            a = Article(text, metaDict)
                            article_word_count = a.getBagOfWords()
                            for word in article_word_count:
                                word_count[word] += article_word_count[word]
                        inText = False
                        text = ''
                elif line.startswith('<REUTERS'):
                    meta = line
                    metaDict = self.readMeta(meta)
        return word_count

    def readResource(self, mode):
        fname = TOPICS_FILE
        if mode == 'terms':
            fname = TERMS_FILE
        resource = []
        resource_dict = {}
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                resource.append(line)
                resource_dict[line] = len(resource) - 1
        if mode == 'topics':
            self.topics_index = resource_dict
            print len(self.topics_index)
        elif mode == 'terms':
            self.terms_index = resource_dict
            print len(self.terms_index)

    def readDir(self, dirName, mode='read'):
        train_topics = set()
        test_topics = set()
        word_count = collections.defaultdict(int)
        count = 0
        for fname in os.listdir(dirName):
            # limit for toy experiments
            #if count > 1:
            #    break
            print 'count', count
            if fname.endswith('.sgm'):
                count += 1
                if mode == 'read':
                    self.readSgm('%s/%s' %(dirName, fname))
                elif mode == 'topics':
                    new_train_topics, new_test_topics = self.learnTopics('%s/%s' %(dirName, fname))
                    train_topics.update(new_train_topics)
                    test_topics.update(new_test_topics)
                elif mode == 'terms':
                    file_word_count = self.learnTerms('%s/%s' %(dirName, fname))
                    for word in file_word_count:
                        word_count[word] += file_word_count[word]
                print '***%d files processed***' % count
                if mode == 'topics':
                    print '%d train topics found' % len(train_topics)
                    print '%d test topics found' % len(test_topics)
                elif mode == 'terms':
                    print '%d terms found' % len(word_count)
        if mode == 'topics':
            final_topics = train_topics.intersection(test_topics)
            with open(TOPICS_FILE, 'w') as f:
                for topic in final_topics:
                    f.write(topic + '\n')
        elif mode == 'terms':
            with open(TERMS_FILE, 'w') as f:
                for word in word_count:
                    if word_count[word] >= 3:
                        f.write(word + '\n')

    def readMeta(self, meta):
        meta = meta.replace('>', '')
        metaFields = [item.split('=') for item in meta.split() if len(item.split('=')) > 1]
        metaDict = {field[0].lower():eval(field[1]) for field in metaFields}
        if self.isModApte(metaDict):
            return metaDict
        return False

    def isModApte(self, metaDict):
        return metaDict['topics'] == 'YES' and metaDict['lewissplit'] != 'NOT-USED'

    def getArticleVector(self, i):
        return self.getArticle(i).getVector(self.terms_index, self.topics_index)

    def buildAndSaveDesignMatrix(self, train_filename, test_filename, meta_filename):
        print 'preparing . . .'

        #N = len(self.articles) # number of articles
        D = len(self.terms_index) # number of terms
        K = len(self.topics_index) # number of topics

        article_ids = sorted(self.articles.keys())
        train_ids = [ID for ID in article_ids if self.articles[ID].split == 'TRAIN']
        test_ids = [ID for ID in article_ids if self.articles[ID].split == 'TEST']
        num_train = len(train_ids)
        num_test = len(test_ids)

        # train data
        print 'creating train data . . .'
        train_data = np.empty((num_train, D + K))

        for n in xrange(num_train):
            train_data[n] = self.getArticleVector(train_ids[n])

        X_train, Y_train = train_data[:,:D], train_data[:,-K:]

        # calculate 'tfc' weights
        # same idf will be used for both train and test sets
        idf = np.ones(D)
        for n in xrange(num_train):
            for d in xrange(D):
                if X_train[n,d] != 0.0:
                    idf[d] += 1.0
        idf = np.log(num_train / idf)

        X_train = np.multiply(X_train, idf)
        skp.normalize(X_train, copy=False)
        train_data = np.hstack((X_train, Y_train))
        np.save(train_filename, train_data)

        # delete once saved to free up memory
        del train_data, X_train, Y_train

        # now test data
        print 'creating test data . . .'
        test_data = np.empty((num_test, D + K))

        for n in xrange(num_test):
            test_data[n] = self.getArticleVector(test_ids[n])

        X_test, Y_test = test_data[:,:D], test_data[:,-K:]

        X_test = np.multiply(X_test, idf)
        skp.normalize(X_test, copy=False)
        test_data = np.hstack((X_test, Y_test))
        np.save(test_filename, test_data)

        # finally save meta data
        print 'saving metadata . . .'
        meta_string = 'dimensions: %d\nclasses: %d\n' % (D, K)
        with open(meta_filename, 'w') as f:
            f.write(meta_string)
        

def main():
    d = Dossier(config.REUTERS_DIR)
    #d.readDir(config.REUTERS_DIR, 'topics')
    #d.readDir(config.REUTERS_DIR, 'terms')
    #d.readDir(config.REUTERS_DIR)
    #print d.getArticle(1)
    #d.buildDesignMatrix()
    #return d
    d.buildAndSaveDesignMatrix(config.REUTERS_TRAIN,
                               config.REUTERS_TEST,
                               config.REUTERS_META)

    
if __name__=="__main__":
    main()
