import pandas
import numpy as np
from nltk import ngrams, pos_tag 
from nltk.corpus import sentiwordnet as swn
from senticnet.senticnet import SenticNet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix, vstack

def pos_tags_tokenizer(tweet):
    tokens = pos_tag(tweet)
    tokens = [tag[1] for tag in tokens]
    return tokens

pos_tags = CountVectorizer(
                tokenizer=pos_tags_tokenizer,
                analyzer="word",
           )

char_ngram = CountVectorizer(
                strip_accents="unicode",
                ngram_range=(1, 3),
                analyzer="char",
             )   

word_ngram = CountVectorizer(
                strip_accents="unicode",
                stop_words="english",
                ngram_range=(1, 3),
                analyzer="word",
             )

vocab_ngram = CountVectorizer(
                strip_accents="unicode",
                stop_words="english",
                ngram_range=(1, 1),
                analyzer="word",
             )

class Sentiment140(BaseEstimator, TransformerMixin):
    def __init__(self, vocab):
        self.csv = pandas.read_csv("data/Sentiment140-Lexicon-v0.1.csv",
                                    lineterminator='\n')
        self.vocab = vocab
        self.x_min = -5.0
        self.x_max = 5.0
        self.X_width = len(self.vocab)

    def fit(self, X):
        return self

    def normalize(self, X):
        ''' Takes input from range [-5,5] and normalizes to range of [-1,1] '''
        X = 2 * ((X - self.x_min) / (self.x_max - self.x_min)) + -1
        return X

    def vector(self, X):
        X = X.split(' ')
        zeros = csr_matrix((1, self.X_width))
        for word in range(len(X)):
            score = self.csv[self.csv["Words"] == X[word]]
            if (not score.empty) and (X[word] in self.vocab):
                if score.size > 1:
                    zeros[0, self.vocab[X[word]]] = np.mean(score["Sentiment"].astype("float32"))
                else:
                    zeros[0, self.vocab[X[word]]] = score["Sentiment"].astype("float32")[0]
                zeros[0, self.vocab[X[word]]] = self.normalize(zeros[0, self.vocab[X[word]]])
        return zeros

    def transform(self, X):
        self.zeros = csr_matrix((0, self.X_width))
        self.X_length = X.shape[0]
        for i in range(self.X_length):
            self.zeros = vstack([self.zeros, self.vector(X[i])])
        return self.zeros

    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform(X)
        return self.zeros

class BingLiu(BaseEstimator, TransformerMixin):
    def __init__(self, vocab):
        self.csv = pandas.read_csv("data/Bing-Liu.csv",
                                    lineterminator='\n')
        self.vocab = vocab
        self.X_width = len(self.vocab)

    def fit(self, X):
        return self
        
    def vector(self, X):
        X = X.split(' ')
        zeros = csr_matrix((1, self.X_width))
        for word in range(len(X)):
            score = self.csv[self.csv["Words"] == X[word]]
            if (not score.empty) and (X[word] in self.vocab):
                if score.size > 1:
                    zeros[0, self.vocab[X[word]]] = np.mean(score["Sentiment"].astype("float32"))
                else:
                    zeros[0, self.vocab[X[word]]] = score["Sentiment"].astype("float32")[0]
        return zeros

    def transform(self, X):
        self.zeros = csr_matrix((0, self.X_width))
        self.X_length = X.shape[0]
        for i in range(self.X_length):
            self.zeros = vstack([self.zeros, self.vector(X[i])])
        return self.zeros
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform(X)
        return self.zeros

class NrcEmotion(BaseEstimator, TransformerMixin):
    def __init__(self, vocab):
        self.csv = pandas.read_csv("data/NRC-Emotion.csv",
                                    lineterminator='\n')
        self.vocab = vocab
        self.X_width = len(self.vocab)

    def fit(self, X):
        return self

    def vector(self, X):
        X = X.split(' ')
        zeros = csr_matrix((1, self.X_width))
        for word in range(len(X)):
            score = self.csv[self.csv["word"] == X[word]]
            if (not score.empty) and (X[word] in self.vocab):
                if score.size > 1:
                    zeros[0, self.vocab[X[word]]] = np.mean(score["emotion-intensity-score"].astype("float32"))
                else:
                    zeros[0, self.vocab[X[word]]] = score["emotion-intensity-score"].astype("float32")[0]
        return zeros

    def transform(self, X):
        self.zeros = csr_matrix((0, self.X_width))
        self.X_length = X.shape[0]
        for i in range(self.X_length):
            self.zeros = vstack([self.zeros, self.vector(X[i])])
        return self.zeros

    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform(X)
        return self.zeros

class NrcHashtag(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X):
        return self
    def transform(self, X):
        return

class SentiWordNet(BaseEstimator, TransformerMixin):
    def __init__(self, vocab):
        self.vocab = vocab
        self.X_width = len(self.vocab)

    def fit(self, X):
        return self

    def vector(self, X):
        X = X.split(' ')
        zeros = csr_matrix((1, self.X_width))
        try:
            ptag = pos_tag(X)
        except IndexError:
            return zeros
        for word in range(len(X)):
            if not X[word] in self.vocab:
                continue
            tag = ptag[word][1]
            if tag.startswith('J'):
                pos = 'a'
            elif tag.startswith('N'):
                pos = 'n'
            elif tag.startswith('R'):
                pos = 'r'
            elif tag.startswith('V'):
                pos = 'v'
            else:
                continue
            score = list(swn.senti_synsets(X[word], pos=pos))
            if score == []:
                continue
            score = score[0]
            zeros[0, self.vocab[X[word]]] = (score.pos_score() - score.neg_score())
        return zeros
            
    def transform(self, X):
        self.zeros = csr_matrix((0, self.X_width))
        self.X_length = X.shape[0]
        for i in range(self.X_length):
            self.zeros = vstack([self.zeros, self.vector(X[i])])
        return self.zeros

    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform(X)
        return self.zeros

class SenticNets(BaseEstimator, TransformerMixin):
    def __init__(self, vocab):
        self.vocab = vocab
        self.X_width = len(vocab)
        self.sn = SenticNet()

    def fit(self, X):
       return self

    def vector(self, X):
        X = X.split(' ')
        zeros = csr_matrix((1, self.X_width))
        for word in range(len(X)):
            if not X[word] in self.vocab:
                continue
            try:
                score = self.sn.polarity_value(X[word])
            except KeyError:
                continue
            zeros[0, self.vocab[X[word]]] = score
        return zeros

    def transform(self, X):
        self.zeros = csr_matrix((0, self.X_width))
        self.X_length = X.shape[0]
        for i in range(self.X_length):
            self.zeros = vstack([self.zeros, self.vector(X[i])])
        return self.zeros

    def fit_transform(self, X, y=None):
        self.fit(X)
        self.transform(X)
        return self.zeros
