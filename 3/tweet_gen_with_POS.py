#!/usr/bin/python
#-*- coding: utf-8 -*-

from nltk.util import ngrams
import string
import random
import nltk
import os


FILE_PATH = 'corpus.txt'

stopwords = ['', '(', ')', '{', '}', '\\', '--', ':', '-', "'s"]
punc = string.punctuation + "``" + "''" + '"'


def tokenized_words(data):
    sents = []
    for sent in nltk.sent_tokenize(data):
        words = [word for word in nltk.word_tokenize(sent) if word not in stopwords and word not in punc]
        sents.append(words)
        
    return sents

def tokenized_rev_words(data):
    sents = []
    for sent in nltk.sent_tokenize(data):
        words = [word for word in nltk.word_tokenize(sent) if word not in stopwords and word not in punc]
        sents.append(list(reversed(words)))
        
    return sents

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

def generate_txt_bigram_model_random(cprob_2gram, cprob_2gram_rev, initialword, numwords=15):
    text = initialword
    suf_word = initialword
    pre_word = initialword
    for index in range(numwords):
        if random.random() > 0.5:
            try:
                suf_word = cprob_2gram[suf_word].generate()
                text = text + " " + suf_word
            except Exception as e:
                print('Can not generate the sentence')
                return
        else:
            try: 
                pre_word = cprob_2gram_rev[pre_word].generate()
                text = pre_word + ' ' + text
            except Exception as e:
                print('Can not generate the sentence')
                return
    return text

# Write code to accept sentences which match the POS template
def filter_sentences(postags_list, sent_list, template):
    filtered_sent = []
    for ind, pos_tag in enumerate(postags_list):
        if is_pos_tag_match(pos_tag, template):
            filtered_sent.append(sent_list[ind])
            
    return filtered_sent
        
        
def is_pos_tag_match(tag, template):
    start = tag[0]
    if start in template:
        for t in tag[1:]:
            if t not in template[start]:
                return False
            else:
                start = t
        else:
            return True
    return False
    
def print_filtered_sent(filt_sent):
    for sent in filt_sent:
        print(sent)

def sent_prob(sent_list, cpd_1gram, cpd_2gram):
    sent_prob_dict = {}
    for sent in sent_list:
        total_prob = 1.0
        words = nltk.word_tokenize(sent)
        total_prob = cpd_1gram.prob(words[0])
        for w1, w2 in nltk.ngrams(words, 2):
            total_prob *= cpd_2gram[w1].prob(w2)
        sent_prob_dict[sent] = total_prob
    return sent_prob_dict

def get_top_five(dict_of_probs):
    sorted_dict = sorted(dict_of_probs.items(), key=lambda x: -x[1])
    top_five_sent = [sent for sent, prob in sorted_dict[:5]]
    return top_five_sent

def main():
    with open(FILE_PATH, 'r') as f:
        data = f.read().lower().replace('\n',' ')

    sents = tokenized_words(data)
    rev_sents = tokenized_rev_words(data)
    train_corpus = [word for sent in sents for word in sent]
    rev_train_corpus = [word for sent in rev_sents for word in sent] 

    cfd_2gram = ngram_freq_dist(train_corpus, 2)
    cfd_2gram_rev = ngram_freq_dist(rev_train_corpus, 2)

    cpd_2gram = nltk.ConditionalProbDist(cfd_2gram, nltk.MLEProbDist)
    cpd_2gram_rev = nltk.ConditionalProbDist(cfd_2gram_rev, nltk.MLEProbDist)

    cfd_1gram = ngram_freq_dist(train_corpus)
    cpd_1gram = nltk.MLEProbDist(cfd_1gram)

    random_sentences = []
    random_pos_tags = []
    random_word_pos_tags = []

    # Generate 200 sentences randomly 
    for _ in range(5000):
        sent = generate_txt_bigram_model_random(cpd_2gram, cpd_2gram_rev, 'education', 9)
        word_pos_tags = nltk.pos_tag(sent.split())
        pos_tags = [x[1] for x in word_pos_tags]
    
        random_word_pos_tags.append(word_pos_tags)
        random_sentences.append(sent)
        random_pos_tags.append(pos_tags)

    '''
    RULES:

    1. Determiner always comes before a noun.
    2. Noun can be followed by another noun phrase.
    3. Modals (could, will) can follow nouns.
    4. ..

    '''

    pos_template_dict = {
        'NN': ['NN', 'VB', 'VBD', 'MD', 'VBP', 'IN', 'VBZ', 'NNS'],
        'NNS': ['NN', 'VB', 'VBD', 'MD', 'VBP', 'IN', 'NN'],
        'NNP': ['NN', 'VB', 'VBD', 'MD', 'VBP', 'IN', 'NNS'],
        'NNPS': ['NN', 'VB', 'VBD', 'MD', 'VBP', 'IN'],
        'DT': ['NN', 'NNS', 'NNP', 'NNPS', 'VBP', 'JJ'],
        'JJ': ['CC'],
        'CC': ['NN', 'NNS', 'NNP', 'NNPS'],
        'VB': ['NN', 'DT', 'TO'],
        'VBD': ['NN', 'TO'],
        'VBG': ['IN', 'TO'],
        'VBP': ['VBG', 'RB', 'TO'],
        'VBN': ['RB', 'PRP', 'TO'],
        'VBZ': ['VBN'],
        'MD': ['VB', 'PRP'],
        'IN': ['DT', 'JJ'],
        'RB': ['NN', 'NNS'],
        'PRP': ['MD', 'VBD'],
        'TO': ['VB'],
    }

    filtered_sent = filter_sentences(random_pos_tags, random_sentences, pos_template_dict)
    #print_filtered_sent(filtered_sent)
    #print('------------------------------------------------------------------------------')

    dict_of_probs = sent_prob(filtered_sent, cpd_1gram, cpd_2gram)
    top_five = get_top_five(dict_of_probs)

    print('Top five tweets:\n')
    for tweet in top_five:
        print(tweet)
        print('=============')

    '''
    Top 5 tweets:

    1. mike pence slashed education must change i 'll be president

    2. us must ensure world-class education than this morning at the

    3. end childhood education should be a household items are often

    4. we must provide education solo es la discriminación y defensora

    5. nation in the education group of the polls are barely

    '''

if __name__ == '__main__':
    main()
