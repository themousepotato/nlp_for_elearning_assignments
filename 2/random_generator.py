#!/usr/bin/python
#-*- coding: utf-8 -*-

import nltk
import os
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter
from nltk.util import ngrams
from math import log

stopwords=['','(',')','{','}','\\','--',':','-',"'s"]
punct=string.punctuation+"``"+"''"+'"'

CORPUS_FILENAME = 'corpus.txt'
REV_CORPUS_FILENAME = 'rev_corpus.txt'

def create_tokens(filename):
    with open(filename, 'r') as f:
        corpus = f.read()

    sents=[]
    for sent in nltk.sent_tokenize(corpus):
        token= [word for word in nltk.word_tokenize(sent) if word not in punct and word not in stopwords]
        sents.append(token)
    return sents

def train_test_split(sents, ratio=0.99):
    if len(sents) > 0:
        spl = int(ratio*len(sents))
        train_corpus = sents[:spl]
        test_corpus = sents[spl:]
    else:
        print('Corpus not created')
    train_corpus = [word for sent in train_corpus for word in sent]
    test_corpus = [word for sent in test_corpus for word in sent]
    return train_corpus, test_corpus

def ngram_freq_dist(corpus, ngram=1):
    if isinstance(corpus, list) and len(corpus)>0:
        train_corpus=corpus
    elif type(corpus) is str:
        train_corpus=nltk.word_tokenize(corpus)
    else:
        print('Error')
        return None
    
    freq_dist=None
    if ngram==1:
        freq_dist = nltk.FreqDist(train_corpus) #freq distibution for unigrams
    elif ngram==2:
        freq_dist = nltk.ConditionalFreqDist(nltk.ngrams(train_corpus, 2))# conditional freq dist for bigrams
    elif ngram==3:
        trigrams_as_bigrams=[]
        trigram =[a for a in ngrams(train_corpus, 3)]
        trigrams_as_bigrams.extend([((t[0],t[1]), t[2]) for t in trigram])
        freq_dist = nltk.ConditionalFreqDist(trigrams_as_bigrams)# conditional freq dist for trigrams
    else:
        print('Supported upto trigrams only')
    return freq_dist

def generate_txt_bigram_model(cprob_2gram, rev_cpd_2gram, initialword, numwords=10):
    prefix_n = random.randint(1, numwords - 2)
    suffix_n = numwords - prefix_n - 1
    prefix = initialword
    suffix = initialword
    suffix_text = initialword
    prefix_text = ''
    for index in range(prefix_n):
        try:
            prefix = rev_cpd_2gram[prefix].generate()
            prefix_text = prefix + ' ' + prefix_text
        except Exception as e:
            print(e)
            return

    prefix_text = ' '.join(reversed(prefix_text.split()))
    for index in range(suffix_n):
        try:
            suffix = cprob_2gram[suffix].generate()
            suffix_text = suffix_text + ' ' +suffix
        except Exception as e:
            print(e)
            return
    return prefix_text + ' ' + suffix_text

def main():
    sents = create_tokens(CORPUS_FILENAME)
    train_corpus, test_corpus = train_test_split(sents)

    cfd_2gram = ngram_freq_dist(train_corpus, ngram=2) #conditional frequency distribution for bigrams
    cpd_2gram = nltk.ConditionalProbDist(cfd_2gram, nltk.MLEProbDist) # conditional probality distribution for bigrams

    rev_sents = create_tokens(REV_CORPUS_FILENAME)
    rev_train_corpus, rev_test_corpus = train_test_split(rev_sents)
    rev_cfd_2gram = ngram_freq_dist(rev_train_corpus, ngram=2)
    rev_cpd_2gram = nltk.ConditionalProbDist(rev_cfd_2gram, nltk.MLEProbDist)

    text_wiki = generate_txt_bigram_model(cpd_2gram, rev_cpd_2gram, 'trump', numwords=10)
    print('Test sentance for trump:', text_wiki)

if __name__ == '__main__':
    main()
