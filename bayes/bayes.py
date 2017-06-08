#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from nltk import stem
import pdb
from sklearn.feature_extraction.text import CountVectorizer

class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for additive smoothing
        self._feat_prob = [] # do not change the name of these vars
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []
        self._cls = []

    def train(self, X, y):
        alpha_smooth = self._smooth

        cls, samples_per_cls = np.unique(y, return_counts=True)
        self._cls = cls
        samples_per_cls = dict(zip(cls,samples_per_cls))

        #Nfeat is number of features (vocab length).. X.shape[0] give # samples
        Ncls, Nfeat = len(cls), X.shape[1]
        self._Ncls, self._Nfeat = Ncls, Nfeat
        num_samples = X.shape[0]

        # _feat_prob is a (2 x len(vocab) ) array of 0s
        # this should hold all the probabilities of different words.
        self._feat_prob = np.zeros((Ncls, Nfeat))

        # this should hold the probability of negative or positive sentiment
        self._class_prob = np.zeros(Ncls)

        # store samples for each class in dict
        split_samples = {}

        # now split up the examples into their separate classes
        for i in range(len(cls)):
            split_samples[cls[i]] = np.zeros((samples_per_cls[cls[i]], Nfeat))
            cur_cls_index = 0
            for j in range(len(X)):
                if y[j] == cls[i]:
                    split_samples[cls[i]][cur_cls_index] = X[j]
                    cur_cls_index += 1

        # this will hold counts for each word in given class
        counts_per_class = {}

        # calculate feature probabilites and class probabilities
        for cl_idx, cl in enumerate(split_samples):
            counts_per_class[cl] = np.sum(split_samples[cl], axis=0) + alpha_smooth
            self._feat_prob[cl_idx] = counts_per_class[cl] / (samples_per_cls[cl] + (alpha_smooth * Nfeat))

            self._class_prob[cl_idx] = float(samples_per_cls[cl]) / num_samples

    def predict(self, X):
        pred = np.zeros(len(X))
        class_prob = np.ones(self._Ncls)

        for i in range(len(X)):
            sample = X[i]
            for j in range(self._Ncls):
                class_prob[j] = self._class_prob[j]
                for feat_idx, feat in enumerate(sample):
                    if feat == 0:
                        class_prob[j] *= (1-self._feat_prob[j][feat_idx])
                    else:
                        class_prob[j] *= (self._feat_prob[j][feat_idx])

            pred[i] = self._cls[class_prob.argmax()]

        return pred

    @property
    def probs(self):
        # please leave this intact, we will use it for marking
        return self._class_prob, self._feat_prob

with open('sentiment_data/rt-polarity_utf8.neg', 'r') as f:
    lines_neg = f.read().splitlines()

with open('sentiment_data/rt-polarity_utf8.pos', 'r') as f:
    lines_pos = f.read().splitlines()

# 1 means negative, 0 means positive.
data_train = lines_neg[0:5000] + lines_pos[0:5000]
data_test = lines_neg[5000:] + lines_pos[5000:]

stemmer = stem.PorterStemmer()
for i in range(len(data_train)):
    stemmed_line = []
    for word in data_train[i].split(" "):
        word = stemmer.stem(word.decode("utf-8"))
        stemmed_line.append(word)
    data_train[i] = " ".join(stemmed_line)

for i in range(len(data_test)):
    stemmed_line = []
    for word in data_test[i].split(" "):
        word = stemmer.stem(word.decode("utf-8"))
        stemmed_line.append(word)
    data_test[i] = " ".join(stemmed_line)

y_train = np.append(np.ones((1,5000)), (np.zeros((1,5000))))
y_test = np.append(np.ones((1,331)), np.zeros((1,331)))

vectorizer = CountVectorizer(lowercase=True, stop_words='english', max_df=1.0, min_df=1, max_features=None, binary=True)

X_train = vectorizer.fit_transform(data_train).toarray()
X_test = vectorizer.transform(data_test).toarray()

clf = MyBayesClassifier(1);
clf.train(X_train,y_train);
y_pred = clf.predict(X_test)
print np.mean((y_test-y_pred)==0)
