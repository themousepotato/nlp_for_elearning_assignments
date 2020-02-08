#!/usr/bin/python
#-*- coding: utf-8 -*-

from language_model import *

def perform_smoothing(technique):
    sents = create_tokens(CORPUS_FILENAME)
    train_corpus, test_corpus=train_test_split(sents)

    fd_1gram = ngram_freq_dist(train_corpus, ngram=1)
    vocab = len(fd_1gram) #vocabulary size

    cpd_1gram_with_smt = technique(fd_1gram) #Obtaining probality distribution with smt 
    cfd_2gram = ngram_freq_dist(train_corpus, ngram=2)
    cpd_2gram_with_smt = nltk.ConditionalProbDist(cfd_2gram, technique, bins=pow(vocab,2))
    cfd_3gram = ngram_freq_dist(train_corpus, ngram=3)
    cpd_3gram_with_smt = nltk.ConditionalProbDist(cfd_3gram, technique, bins=pow(vocab,3))

    pws_2gram = probable_words('donald trump', cpd_2gram_with_smt, 2)
    pws_3gram = probable_words('donald trump', cpd_3gram_with_smt, 3)
    print('Probable words for donald trump using 3 gram model', pws_3gram)

    test_sent1 = 'donald president is trump'
    test_sent2 = 'donald trump is president'

    prob_1gram = find_sent_prob(test_sent2, cpd_1gram_with_smt)
    print('Sentance probability of {} using 1-gram'.format(test_sent2), prob_1gram)

    prob_2gram = find_sent_prob(test_sent2, cpd_1gram_with_smt, cpd_2gram_with_smt, ngram=2)
    print('Sentance probability of {} using 2-gram'.format(test_sent2), prob_2gram)

    prob_3gram = find_sent_prob(test_sent1, cpd_1gram_with_smt, cpd_2gram_with_smt, cpd_3gram_with_smt, ngram=3)
    print('Sentance probability of {} using 3-gram'.format(test_sent1), prob_3gram)

    print('Entropy of 1 gram model', entropy(cpd_1gram_with_smt, test_corpus, 1))
    print('Entropy of 2 gram model', entropy(cpd_2gram_with_smt, test_corpus, 2))
    print('Entropy of 3 gram model', entropy(cpd_3gram_with_smt, test_corpus, 3))

    print('Perplexity of 1 gram model', perplexity(cpd_1gram_with_smt, test_corpus, 1))
    print('Perplexity of 2 gram model', perplexity(cpd_2gram_with_smt, test_corpus, 2))
    print('Perplexity of 3 gram model', perplexity(cpd_3gram_with_smt, test_corpus, 3))

    text_wiki = generate_txt_bigram_model(cpd_2gram_with_smt, 'trump', numwords=10)
    print('Test sentance for trump:', text_wiki)

def perform_kn_smoothing():
    def ngram_freq_dist(corpus):
        freq_dist = freq_3gram = nltk.FreqDist(ngrams(corpus,3))
        return freq_dist

    def trigram_prob(w1, w2, w3, prob_3gram):
        if prob_3gram is not None:
            #cprob_3gram=nltk.ConditionalProbDist(cfreq_3gram, nltk.MLEProbDist)
            prob=prob_3gram.prob((w1, w2, w3))
        else:
            print('Probablity distribution is not available')
        return prob

    def entropy(pd, text):
        ngram=3
        test_corpus = text
        e = 0.0
        for i in range(ngram - 1, len(test_corpus)):
            context = test_corpus[i - ngram + 1:i]
            word = test_corpus[i]
            #print(str(context)+"    "+token)
            e += logprob(word, context, pd)
        return e / float(len(text) - (ngram - 1))

    def logprob(word, context, pd):
        p=trigram_prob(context[0], context[1], word, pd)
        try:
            val=-p*log(p , 2)
        except ValueError:
            val=0.0
        return val

    def perplexity(pd, text):
        test_corpus=text
        ent=entropy(pd, text)
        #print(ent)
        return pow(2.0, ent)

    def find_sent_prob(sent, pd):
        total_prob=1
        for w1, w2, w3 in nltk.ngrams(nltk.word_tokenize(sent),3):
            #print(w1,w2,w3,trigram_prob(w1, w2, w3, pd))
            total_prob *= (trigram_prob(w1, w2, w3, pd))
        return total_prob

    sents = create_tokens(CORPUS_FILENAME)
    train_corpus, test_corpus = train_test_split(sents)

    fd_3gram = ngram_freq_dist(train_corpus)
    prob_3gram_kn = nltk.KneserNeyProbDist(fd_3gram) #probality distribution with Kneser ney smoothing

    test_sent2 = 'donald trump is president'

    print('Entropy of 3 gram model', entropy(prob_3gram_kn, test_corpus))
    print('Perplexity of 3 gram model', perplexity(prob_3gram_kn, test_corpus))
    print('Sentance probability of {}'.format(test_sent2), find_sent_prob(test_sent2, prob_3gram_kn))


if __name__ == '__main__':
    print('Laplace smoothing\n-----')
    perform_smoothing(nltk.LaplaceProbDist)
    print('Good-Turing smoothing\n-----')
    perform_smoothing(nltk.SimpleGoodTuringProbDist)
    print('KneserNey smoothing\n-----')
    perform_kn_smoothing()