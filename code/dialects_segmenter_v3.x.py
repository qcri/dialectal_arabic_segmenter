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
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import codecs
import json
from time import gmtime, strftime
import datetime
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from ChainCRF import ChainCRF

np.random.seed(1337)  # for reproducibility
#sys.stdin = codecs.getreader('utf-8')(sys.stdin)

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
    return re.sub(r'[ـًٌٍَُِّْ]', '', instring)

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
    for aword in re.split(r'،|٫|٫|٬|؛',txtstring):
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

                elements.append(word)
            else:
                for elt in word_tokenize(word):
                    elements.append(elt)
    output = ''
    for elt in elements:
        output = output + ' ' + elt 
    return output


def getLabels(filepathT,filepathD):
    labels = []
    for line in open(filepathT):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    for line in open(filepathD):
        line = line.strip()
        if len(line) == 0:
            continue
        splits = line.split('\t')
        labels.append(splits[1].strip())
    return list(set(labels))


def load_data(options,seg_tags):
    X_words_train, y_train = load_file(options.train)
    X_words_dev, y_dev = load_file(options.dev)

    index2word = _fit_term_index(X_words_train+X_words_dev, reserved=['<PAD>', '<UNK>'])
    word2index = _invert_index(index2word)

    index2pos = seg_tags
    pos2index = _invert_index(index2pos)

    X_words_train = np.array([[word2index[w] for w in words] for words in X_words_train])
    y_train = np.array([[pos2index[t] for t in s_tags] for s_tags in y_train])

    X_words_dev = np.array([[word2index[w] for w in words] for words in X_words_dev])
    y_dev = np.array([[pos2index[t] for t in s_tags] for s_tags in y_dev])

    return (X_words_train, y_train), (X_words_dev, y_dev),(index2word, index2pos),word2index


def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}

def build_words(src,ref,pred):
    words = []
    rtags = []
    ptags = []
    w = ''
    r = ''
    p = ''
    for i in range(len(src)):
        if(src[i] == 'WB'):
            words.append(w)
            rtags.append(r)
            ptags.append(p)
            w = ''
            r = ''
            p = ''
        else:
            w += src[i]
            r += ref[i]
            p += pred[i]           

    return words, rtags, ptags

def load_file(path):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(path):
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
    words, tags = zip(*[zip(*row) for row in sentences])
    return words, tags

def load_lookuplist(path):
    """
    Load lookp list.
    """
    listwords = {}
    for line in open(path):
        line = line.rstrip()
        listwords[line.replace('+','')] = line     
    return listwords


def build_model(model_path, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim):

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

    #model.load_weights(model_path + '/seg_keras_weights.hdf5')

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--train",  default= "data/joint.trian.3", help="Train set location")
    parser.add_argument("-d", "--dev",  default= "data/joint.dev.3", help="Dev set location")
    parser.add_argument("-s", "--test",   default="", help="Test set location")
    parser.add_argument("-m", "--model",  default="models", help="Model location")
    parser.add_argument("-l", "--lookup",  default= "data/lookup_list.txt", help="Lookup list location")
    parser.add_argument("-p", "--epochs",  default=100, type=int, help="Lookup list location")
    parser.add_argument("-i", "--input",  default="", help="Input stdin or file")
    parser.add_argument("-o", "--output", default="", help="output file")
    parser.add_argument("-k","--task", choices=['train','evaluate','decode'], help="Choice for task (train, evaluate, decode)")

    options = parser.parse_args()

    #print("Task:",options.task)

    #assert os.path.isfile(options.train)
    #assert os.path.exists(options.model)
    #assert os.path.isfile(options.test)

    #seg_tags = getLabels(options.train)
    #label2Idx = dict((l, i) for i, l in enumerate(seg_tags))
    #idx2Label = dict((i, l) for i, l in enumerate(seg_tags))
    seg_tags = ['E', 'S', 'B', 'M', 'WB'] #['WB', 'S', 'B', 'E', 'M']
    idx2Label = {0:'E', 1:'S', 2:'B', 3:'M', 4:'WB'}
    label2Idx = {'E':0, 'S':1, 'B':2, 'M':3, 'WB':4}

    word_embedding_dim = 200
    lstm_dim = 200

    #print('Loading data...')
    (X_words_train,y_train), (X_words_dev,y_dev), (index2word, index2pos),word2index = load_data(options,seg_tags)

    seq_len = []
    for i in range(len(X_words_train)):
        seq_len.append(len(X_words_train[i]))
    #print("MaxLen Train:",max(seq_len))

    maxlen = max(seq_len)  # cut texts after this number of words (among top max_features most common words)
    maxlen = 500 # Set to 500 max num of chars in one line.

    result = map(len,X_words_train)

    max_features = len(index2word)
    nb_seg_tags = len(index2pos)

    eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading Lookup list....')
    lookupList = load_lookuplist(options.lookup)

    
    if(options.task == 'train'):


        X_words_train = sequence.pad_sequences(X_words_train, maxlen=maxlen, padding='post')
        y_train = sequence.pad_sequences(y_train, maxlen=maxlen, padding='post')
        y_train = np.expand_dims(y_train, -1)

        X_words_dev = sequence.pad_sequences(X_words_dev, maxlen=maxlen, padding='post')
        y_dev = sequence.pad_sequences(y_dev, maxlen=maxlen, padding='post')
        y_dev = np.expand_dims(y_dev, -1)

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Build model...')
        model = build_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)

        early_stopping = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.hdf5',verbose=1,save_best_only=True)

        model_json = model.to_json()
        with open(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.json', 'w') as json_file:
            json_file.write(model_json)
        print("saved json")
        model.fit(x=X_words_train, y=y_train,
          validation_data=(X_words_dev, y_dev),
          verbose=1,
          batch_size=64,
          epochs=options.epochs,
          callbacks=[early_stopping, checkpointer])  
        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Save the trained model...')
        # serialize model to JSON
        #
    elif(options.task == 'decode'):

        model = build_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)
        #model.load_weights(options.model + '/'+'seg_keras_weights.hdf5')
        model.load_weights(options.model + '/'+'joint.trian.3_keras_weights.hdf5')

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
            if(idx2Label[test_y_pred[i]] in ('B','M')):
                segt += in_data[i]
                word += listchars[i]
            elif(idx2Label[test_y_pred[i]] in ('E','S') and idx2Label.get(test_y_pred[i+1]) !='WB'):
                segt += in_data[i]+'+'
                word += listchars[i]
            elif(idx2Label[test_y_pred[i]] in ('E','S') and idx2Label.get(test_y_pred[i+1]) =='WB'):
                segt += in_data[i]
                word += listchars[i]
            elif(idx2Label[test_y_pred[i]] == 'WB'):
                if(word in lookupList):
                    
                    printf('%s ',lookupList[word])
                else:
                    if('<UNK>' in segt):
                        segt = word
                    
                    printf('%s ',segt)  
                word = ''
                segt = ''            
            if(sentences_len[sent] == i):
                print('')
                sent = sent + 1 
        print('')

    elif(options.task == 'evaluate'):

        model = build_model(options.model, max_features, word_embedding_dim, maxlen, nb_seg_tags, lstm_dim)
        model.load_weights(options.model + '/'+options.train.split('/')[-1]+'_keras_weights.hdf5')

        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Loading input...')
        X_words, y_test_ref = load_file(options.test)
        X_words_test = np.array([[(word2index[w] if w  in word2index else word2index['<UNK>']) for w in words] for words in X_words])
        X_words_test = sequence.pad_sequences(X_words_test, maxlen=maxlen, padding='post')
        #y_test = sequence.pad_sequences(y_test_ref, maxlen=maxlen, padding='post')
        #y_test = np.expand_dims(y_test, -1)


        eprint(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Decoding ...')
        y_test_pred = model.predict(X_words_test, batch_size=200).argmax(-1)[X_words_test > 0]

        preds = [idx2Label[x] for x in y_test_pred]
        srcs  = list(chain.from_iterable(words for words in X_words))
        refs  = list(chain.from_iterable(tags for tags in y_test_ref))

        #refs  = [(tag for tag in tags) for tags in y_test_ref]
        print("Evaluation Seg Characters:")
        print(classification_report(
                refs,
                preds,
                digits=3,
                target_names=list(set(refs+preds))
            )) #.split('\n')[-2:]

        outfile = open(options.test+'.out','w')
        for i in range(len(preds)):
            outfile.write('%s\t%s\t%s\n'%(srcs[i],refs[i],preds[i]))
        outfile.close()

        (words,rtags,ptags) = build_words(srcs,refs,preds)

        print("Evaluation Seg Words:")
        print('\n'.join(classification_report(
                rtags,
                ptags,
                digits=3,
                target_names=list(set(rtags+ptags))
            ).split('\n')[-4:])) #

        outfile = open(options.test+'.words.out','w')
        for i in range(len(words)):
            outfile.write('%s\t%s\t%s\n'%(words[i],rtags[i],ptags[i]))
        outfile.close()

if __name__ == "__main__":
    main()
