#!/usr/bin/env python

from bs4 import BeautifulSoup
import nltk
import re
import os  
import config
import collections

STOPWORDS = nltk.corpus.stopwords.words('english')
STEMMER = nltk.stem.SnowballStemmer('english')

TOPICS_FILE = 'topics.txt'
TERMS_FILE = 'terms.txt'

class Article:
    
    def __init__(self, text, metaDict):
        #self.text = text
        # body tags cannot be accessed by BeautifulSoup
        #soup = BeautifulSoup(self.text.replace("BODY","CONTENT"), 'lxml')
        #self.date = soup.date
        #self.topics = soup.topics
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
        self.text = text
        self.words = self.stemAndRemoveStopwords(text)
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
        if len(text) == 0:
            text = self.text
        text = re.sub('[^a-z]',' ',text.lower())
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            word_list += [STEMMER.stem(w) for w in words if w not in STOPWORDS]
        for word in word_list:
            word_count[word] += 1
        return word_count

class Dossier:
    
    def __init__(self):
        self.articles = {}
    
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
            meta = ''
            text = ''
            metaDict = None
            for line in f:
                if inArticle:
                    text += line
                    if line.startswith("</REUTERS>"):
                        inArticle = False
                        a = Article(text, metaDict)
                        self.addArticle(a)                            
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

    def readDir(self, dirName, mode='read'):
        train_topics = set()
        test_topics = set()
        word_count = collections.defaultdict(int)
        count = 0
        for fname in os.listdir(dirName):
            if fname.endswith('.sgm'):
                count += 1
                if mode == 'read':
                    return self.readSgm('%s/%s' %(dirName, fname))
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

def main():
    d = Dossier()
    #d.readDir(config.REUTERS_DIR, 'topics')
    d.readDir(config.REUTERS_DIR, 'terms')
    #print d.getArticle(1)
    
if __name__=="__main__":
    main()
