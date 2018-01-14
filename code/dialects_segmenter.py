#!/usr/bin/env python
# -*- coding: utf-8 -*-
# this script uses pretrained model to segment Arabic dialect data.
# it takes the pretrained model trained on joint dialects and the 
# training vocab and produces segmented text
#
# Copyright (C) 2017, Qatar Computing Research Institute, HBKU, Qatar
# Las Update: Sun Oct 29 15:34:43 +03 2017
#
# BibTex: @inproceedings{samih2017learning,
#  title={Learning from Relatives: Unified Dialectal Arabic Segmentation},
#  author={Samih, Younes and Eldesouki, Mohamed and Attia, Mohammed and Darwish, Kareem and Abdelali, Ahmed and Mubarak, Hamdy and Kallmeyer, Laura},
#  booktitle={Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)},
#  pages={432--441},
#  year={2017}}
#
from __future__ import print_function

__author__ = 'Ahmed Abdelali (aabdelali@hbku.edu.qa)'


import numpy as np
import sys
import os
import re
import argparse
from collections import Counter
from itertools import chain
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import Callback
#from keras.utils.data_utils import get_file
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import ChainCRF
import codecs
from time import gmtime, strftime
import datetime
from nltk.tokenize import word_tokenize


np.random.seed(1337)  # for reproducibility
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

def printf(format, *args):
    sys.stdout.write(format % args)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def valid_date(datestring):
    try:
        mat=re.match('^(\d{2})[/.-](\d{2})[/.-](\d{4})$', datestring)
        if mat is not None:
            datetime.datetime(*(map(int, mat.groups()[-1::-1])))
            return True
    except ValueError:
        pass
    return False

def valid_number(numstring):
    try:
        mat=re.match("^[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?$", numstring)
        if mat is not None:
            return True
    except ValueError:
        pass
    return False
    
def valide_time(timestring):
    try:
        mat=re.match('^(2[0-3]|[01]?[0-9]):([0-5]?[0-9])$', timestring)
        if mat is not None:
            datetime.time(*(map(int, mat.groups()[::])))
            return True
        mat=re.match('^(2[0-3]|[01]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9])$', timestring)
        if mat is not None:
            datetime.time(*(map(int, mat.groups()[::])))
            return True
        mat=re.match('^(2[0-3]|[01]?[0-9]):([0-5]?[0-9]):([0-5]?[0-9]).([0-9]?[0-9])$', timestring)
        if mat is not None:
            datetime.time(*(map(int, mat.groups()[::])))
            return True
    except ValueError:
        pass        
    return False

def valid_email(emailstring):
    try:
        mat=re.match('^[^@]+@[^@]+\.[^@]+$',emailstring)
        if mat is not None:
            return True
    except ValueError:
        pass
    return False

def removeDiacritics(instring):
    return re.sub(ur'[ـًٌٍَُِّْ]', '', instring)

def isDelimiter(ch):
    if(ord(ch) == 32): #\u0020
        return True
    elif(ord(ch)>=0 and ord(ch)<=47): #\u0000-\u002F
        return True
    elif(ord(ch)>=58 and ord(ch)<=64): #\u003A-\u0040
        return True
    elif(ord(ch)>=123 and ord(ch)<=187): #\u007B-\u00BB
        return True
    elif(ord(ch)>=91 and ord(ch)<=96): #\u005B-\u005D
        return True
    elif(ord(ch)>=1536 and ord(ch)<=1548): #\u0600-\u060C
        return True
    elif(ord(ch)>=1748 and ord(ch)<=1773): #\u06D4-\u06ED
        return True
    elif(ord(ch)==65279): #\ufeff
        return True
    else:
        return False

def tokenizeline(txtstring):
    elements =[]
    #Remove Kashida and diacritics.
    txtstring = removeDiacritics(txtstring)

    #Split on Arabic delimiters
    for aword in re.split(ur'،|٫|٫|٬|؛',txtstring):
        for word in aword.split():
            #print("==>",word)
            if (word.startswith("#")
                or word.startswith("@")
                or word.startswith(":")
                or word.startswith(";")
                or word.startswith("http://")
                or word.startswith("https://")
                or valid_email(word)
                or valid_date(word)
                or valid_number(word)
                or valide_time(word)):

                elements.append(word);
            else:
                for elt in word_tokenize(word):
                    elements.append(elt);
    output = ''
    for elt in elements:
        output = output + ' ' + elt 
    return output


def getLabels(filepath):
    labels = []
    for line in codecs.open(filepath, 'r','utf-8'):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    return list(set(labels))


def load_data(options,seg_tags):
    X_words_train, y_train = load_file(options.train)

    index2word = _fit_term_index(X_words_train, reserved=['<PAD>', '<UNK>'])
    word2index = _invert_index(index2word)

    index2pos = seg_tags
    pos2index = _invert_index(index2pos)

    X_words_train = np.array([[word2index[w] for w in words] for words in X_words_train])
    y_train = np.array([[pos2index[t] for t in s_tags] for s_tags in y_train])

    return (X_words_train, y_train), (index2word, index2pos),word2index


def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}


def load_file(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path,'r','utf-8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    words, pos_tags = zip(*[zip(*row) for row in sentences])
    return words, pos_tags

def load_lookuplist(path):
    """
    Load lookp list.
    """
    listwords = {}
    for line in codecs.open(path,'r','utf-8'):
        line = line.rstrip()
        listwords[line.replace('+','')] = line     
    return listwords


def load_model(model_path, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim):

    model = Sequential()
    model.add(Embedding(max_features, word_embedding_dim, input_length=maxlen, name='word_emb', mask_zero=True))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(lstm_dim,return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(nb_seg_tags)))
    crf = ChainCRF()
    model.add(crf)
    model.compile(loss=crf.sparse_loss,
                   optimizer= RMSprop(0.01),
                   metrics=['sparse_categorical_accuracy'])
    #model.compile('adam', loss=crf.sparse_loss, metrics=['sparse_categorical_accuracy'])
    #early_stopping = EarlyStopping(patience=10, verbose=1)
    #checkpointer = ModelCheckpoint(options.model + "/seg_keras_weights.hdf5",verbose=1,save_best_only=True)
    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading saved model:'+model_path + '/seg_keras_weights.hdf5')

    model.load_weights(model_path + '/seg_keras_weights.hdf5')

    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--train",  default="data/joint.trian.3", help="Train set location")
    parser.add_argument("-m", "--model",  default="models", help="Test set location")
    parser.add_argument("-l", "--lookup",  default="data/lookup_list.txt", help="Lookup list location")
    parser.add_argument("-i", "--input",  default="", help="Input stdin or file")
    parser.add_argument("-t", "--test",   default="", help="Test set location")
    parser.add_argument("-o", "--output", default="", help="output file")

    options = parser.parse_args()

    print("Arg:",options.train)
    assert os.path.isfile(options.train)
    assert os.path.exists(options.model)
    #assert os.path.isfile(options.test)

    seg_tags = getLabels(options.train)
    label2Idx = dict((l, i) for i, l in enumerate(seg_tags))
    idx2Label = dict((i, l) for i, l in enumerate(seg_tags))


    word_embedding_dim = 200
    lstm_dim = 200

    #print('Loading data...')
    (X_words_train,y_train), (index2word, index2pos),word2index = load_data(options,seg_tags)



    seq_len = []
    for i in range(len(X_words_train)):
        seq_len.append(len(X_words_train[i]))
    #print("MaxLen Train:",max(seq_len))

    maxlen = max(seq_len)  # cut texts after this number of words (among top max_features most common words)
    maxlen = 500 # Set to 500 max num of chars in one line.

    result = map(len,X_words_train)

    max_features = len(index2word)
    nb_seg_tags = len(index2pos)

    X_words_train = sequence.pad_sequences(X_words_train, maxlen=maxlen, padding='post')

    y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
    y_train = np.expand_dims(y_train, -1)

    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading Lookup list....')
    lookupList = load_lookuplist(options.lookup)


    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Build model...')
    model = load_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)

    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading input...')
    sentences = []
    sentences_len = []
    l  = 0
    for line in sys.stdin:
        
        sentence = []

        if len(line) < 2:
            print("")
            continue
        words = tokenizeline(line).strip().split()
        for word in words:
                for ch in word:
                    sentence.append([ch,'WB'])
                    l = l + 1
                sentence.append(['WB','WB'])
                l = l + 1
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
                sentences_len.append(l)

    listwords,tags = zip(*[zip(*row) for row in sentences])

    X_words_test = np.array([[word2index.get(w, word2index['<UNK>']) for w in words] for words in listwords])
    X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')

    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Decoding ...')
    test_y_pred = model.predict(X_words_test, batch_size=200).argmax(-1)[X_words_test > 0]

    in_data = []
    for i in range(len(X_words_test)):
        for j in range(len(X_words_test[i])):
            if X_words_test[i][j] > 0:
                in_data.append(index2word[X_words_test[i][j]])

    listchars = []
    for words in listwords:
        for w in words:
            listchars.append(w)

    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Writing output ...')

    word = ''
    segt = ''
    sent = 0
    for i in range(len(test_y_pred)):
        if(idx2Label.get(test_y_pred[i]) in ('B','M')):
            segt += in_data[i]
            word += listchars[i]
        elif(idx2Label.get(test_y_pred[i]) in ('E','S') and idx2Label.get(test_y_pred[i+1]) !='WB'):
            segt += in_data[i]+'+'
            word += listchars[i]
        elif(idx2Label.get(test_y_pred[i]) in ('E','S') and idx2Label.get(test_y_pred[i+1]) =='WB'):
            segt += in_data[i]
            word += listchars[i]
        elif(idx2Label.get(test_y_pred[i]) == 'WB'):
            if(lookupList.has_key(word)):
                
                printf('%s ',lookupList[word].encode('utf-8'))
            else:
                if('<UNK>' in segt):
                    segt = word
                
                printf('%s ',segt.encode('utf-8'))  
            word = ''
            segt = ''            
        if(sentences_len[sent] == i):
            print('')
            sent = sent + 1 
    print('')


if __name__ == "__main__":
    main()
