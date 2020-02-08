#!/usr/bin/python
#-*- coding: utf-8 -*-

import nltk
import os
import string
from nltk.util import ngrams
from math import log
import matplotlib.pyplot as plt

stopwords=['','(',')','{','}','\\','--',':','-',"'s"]
punct=string.punctuation+"``"+"''"+'"'

CORPUS_FILENAME = 'corpus.txt'

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

def cross_validation(sents, k=5):
    inx=0
    fold_size=int(len(sents)/k)
    for i in range(1, k+1):
        test_sents=sents[inx:inx+fold_size]
        train_sents=sents[0:inx]+sents[inx+fold_size:]
        inx=i*fold_size
        test_sents = [word for sent in test_sents for word in sent]
        train_sents = [word for sent in train_sents for word in sent]
        yield train_sents, test_sents

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

def unigram_prob(word, cpd_1gram):
    if cpd_1gram is not None:
        return cpd_1gram.prob(word)
    else:
        print('Probablity distribution is not available')
    
def conditional_bigram_prob(word1, word2, cprob_2gram):
    if cprob_2gram is not None:
        #cprob_2gram = nltk.ConditionalProbDist(cfreq_2gram, nltk.MLEProbDist)
        cpd=cprob_2gram[word1].prob(word2)
    else:
        print('Probablity distribution is not available')
    return cpd

def conditional_trigram_prob(w1, w2, w3, cprob_3gram):
    if cprob_3gram is not None:
        #cprob_3gram=nltk.ConditionalProbDist(cfreq_3gram, nltk.MLEProbDist)
        cprob_3gram=cprob_3gram[(w1, w2)].prob(w3)
    else:
        print('Probablity distribution is not available')
    return cprob_3gram

def probable_words(sequence, cpd, ngram=2, num=10):
    context=nltk.word_tokenize(sequence)
    temp=[]
    if ngram==2 and len(cpd) > 0:
        context_word=context[-1:]
        #cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
        fd=cpd[context_word[0]].freqdist() # Gives dictionary which shows following words and their frequency given 
        # context_word[0]
        if len(fd) > 0:
            for i, item in enumerate(reversed(sorted(fd.items(), key=lambda item: item[1]))):
                temp.append(item)
                if i > num:
                    break
        else:
            print('No probable words available in corpus following the last word')
    elif ngram==3 and len(cpd) > 0:
        context_word=context[-2:]
        #cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
        fd=cpd[(context_word[0],context_word[1])].freqdist() 
        if len(fd) > 0:
            for i, item in enumerate(reversed(sorted(fd.items(), key=lambda item: item[1]))):
                temp.append(item)
                if i > num:
                    break
        else:
            print('No probable words available in corpus following the last word')
        
    else:
        print('Support only for bigrams and trigrams')
    if len(temp) > 0:
        probable_word_plot(temp, " ".join(context_word), num)
    return temp

def probable_word_plot(freq_dist, context='', num=10):
    words=[]
    values=[]
    if len(freq_dist) > num:
        for a in freq_dist[:num]:
            words.append(a[0])
            values.append(a[1])
    else:
        for a in freq_dist:
            words.append(a[0])
            values.append(a[1])
    plt.rcParams['figure.figsize'] = (15,6)
    plt.rcParams['font.size'] = 10
    plt.bar(words, values, width=0.10)
    plt.title('Probable Words given context: "{0}"'.format(context))
    plt.xlabel('Probable Words')
    plt.ylabel('Frequency')
    plt.show()
    return

def find_sent_prob(seq, pd1=None, pd2=None, pd3=None, ngram=1):
    words=nltk.word_tokenize(seq)
    total_prob=1.0
    if ngram==1 and pd1 is not None:
        for word in nltk.word_tokenize(seq):
            total_prob *= (unigram_prob(word, pd1))
        return total_prob
    elif ngram==2 and pd1 is not None and pd2 is not None:
        total_prob *= unigram_prob(words[0], pd1)
        for w1, w2 in nltk.ngrams(words, 2):
            total_prob *= (conditional_bigram_prob(w1, w2, pd2))
    elif ngram==3 and pd1 is not None and pd2 is not None and pd3 is not None:
        total_prob *= unigram_prob(words[0], pd1) * conditional_bigram_prob(words[0], words[1], pd2)
        for w1, w2, w3 in nltk.ngrams(words,3):
            total_prob *= (conditional_trigram_prob(w1, w2, w3, pd3))
    else:
        print('Check the arguments')
        total_prob=0.0
    return total_prob

def generate_txt_bigram_model(cprob_2gram, initialword, numwords=10):
    #cprob_2gram = nltk.ConditionalProbDist(cfreq_2gram, nltk.MLEProbDist)
    text=initialword
    word=initialword
    for index in range(numwords):
        try:
            word = cprob_2gram[word].generate()
            text=text +" "+word
        except Exception as e:
            print('Can not generate the sentence')
            return
    return text

def generate_txt_trigram_model(cprob_3gram, initialword1, initialword2, numwords=10):
    #cprob_3gram = nltk.ConditionalProbDist(cfreq_3gram, nltk.MLEProbDist)
    text=initialword1+" "+initialword2
    word1=initialword1
    word2=initialword2
    for index in range(numwords):
        try:
            word3=cprob_3gram[(word1, word2)].generate()
            text=text +" "+word3
            word1=word2
            word2=word3
        except Exception as e:
            print('Can not generate the sentence')
            return
    return text


def entropy(pd, text, ngram):
    test_corpus = text
    e = 0.0
    for i in range(ngram - 1, len(test_corpus)):
        context = test_corpus[i - ngram + 1:i]
        word = test_corpus[i]
        #print(str(context)+"    "+token)
        e += logprob(word, context, pd)
    #retrun e
    return e / float(len(text) - (ngram - 1))

def logprob(word, context, pd):
    if len(context)==0:
        p=unigram_prob(word,pd)
    elif len(context)==1:
        p=conditional_bigram_prob(context[0], word, pd)
    else:
        p=conditional_trigram_prob(context[0], context[1], word, pd)
    try:
        val=-p*log(p , 2)
    except ValueError:
        val=0.0
    return val

def perplexity(pd, text, ngram):
    test_corpus=text
    ent=entropy(pd, text, ngram)
    #print(ent)
    return pow(2.0, ent)

def main():
    sents = create_tokens(CORPUS_FILENAME)
    train_corpus, test_corpus = train_test_split(sents)

    fd_1gram = ngram_freq_dist(train_corpus, ngram=1)
    cpd_1gram = nltk.MLEProbDist(fd_1gram)

    freq_dist2 = ngram_freq_dist(train_corpus, 2)

    print('Nations', freq_dist2["nations"])

    cfd_2gram = ngram_freq_dist(train_corpus, ngram=2) #conditional frequency distribution for bigrams
    cpd_2gram = nltk.ConditionalProbDist(cfd_2gram, nltk.MLEProbDist) # conditional probality distribution for bigrams

    cfd_3gram = ngram_freq_dist(train_corpus, ngram=3)
    cpd_3gram = nltk.ConditionalProbDist(cfd_3gram, nltk.MLEProbDist)

    pws_2gram = probable_words('united states', cpd_2gram, 2)
    pws_3gram = probable_words('donald trump', cpd_3gram, 3)

    print('Probable words for donald trump using 3 gram model', pws_3gram)

    test_sent1 = 'donald president is trump'
    test_sent2 = 'donald trump is president'

    prob_1gram = find_sent_prob(test_sent2, cpd_1gram, ngram=1)
    print('Sentance probability of {}'.format(test_sent1), prob_1gram)

    print('Entropy of 1 gram model', entropy(cpd_1gram, test_corpus, 1))
    print('Entropy of 2 gram model', entropy(cpd_2gram, test_corpus, 2))
    print('Entropy of 3 gram model', entropy(cpd_3gram, test_corpus, 3))

    print('Perplexity of 1 gram model', perplexity(cpd_1gram, test_corpus, 1))
    print('Perplexity of 2 gram model', perplexity(cpd_2gram, test_corpus, 2))
    print('Perplexity of 3 gram model', perplexity(cpd_3gram, test_corpus, 3))

    text_wiki = generate_txt_bigram_model(cpd_2gram, 'trump', numwords=10)
    print('Test sentance for trump:', text_wiki)

if __name__ == '__main__':
    main()