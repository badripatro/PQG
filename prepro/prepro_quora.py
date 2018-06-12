"""
Preprocesses a raw json dataset into hdf5/json files.
python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
To get the question features. You will also see some question statistics in the terminal output. This will generate two files in the current directory, only_quora_tt.h5 and only_quora_tt.json.
Also in this code a lot of places you will find things related to captions, but they actually correspond to paraphrases. Reuse of previous code :p
"""

import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json

import re
import math


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, params):
  
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(s)
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def prepro_question1(imgs, params):
  
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question1']
        if params['token_method'] == 'nltk':
            txt_c = word_tokenize(s)
        else:
            txt_c = tokenize(s)

        img['processed_tokens_caption'] = txt_c #this name is a bit misleading, it is for paraphrase questions actually.
        if i < 10: print txt_c
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs


def build_vocab_question(imgs5, params):#imgs1,imgs2,imgs3,imgs4,imgs5,imgs6,imgs7,imgs8,
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
        
    for img in imgs5:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]   # will incorpate vocab for  both caption and question
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)


    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
    
    for img in imgs5:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_caption']
        caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt_c]
        img['final_caption'] = caption
    
    return imgs5,vocab#, imgs1,imgs2,imgs3,imgs4,imgs5,imgs6,imgs7,imgs8, vocab

def apply_vocab_question(imgs, wtoi):  ## this is for val or test question and caption 
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_caption']
        caption = [w if w in wtoi else 'UNK' for w in txt_c]
        img['final_caption'] = caption  

    return imgs

def encode_question2(imgs, params, wtoi):

    max_length = params['max_length']
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    
    caption_arrays = np.zeros((N, max_length), dtype='uint32') # will store encoding caption words
    caption_length = np.zeros(N, dtype='uint32')# will store encoding caption words
       
    
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['id'] #unique_id
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this question sequence
        caption_length[question_counter] = min(max_length, len(img['final_caption'])) # record the length of this caption sequence        
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
        for k,w in enumerate(img['final_caption']):         ## this is for caption
            if k < max_length:
                caption_arrays[i,k] = wtoi[w]            
  
    return label_arrays, label_length, question_id, caption_arrays, caption_length


def main(params):
    
    imgs_train5 = json.load(open(params['input_train_json5'], 'r'))
    imgs_test5 = json.load(open(params['input_test_json5'], 'r'))

    
    
    ##seed(123) # make reproducible
    ##shuffle(imgs_train) # shuffle the order

    
    # tokenization and preprocessing training question
    imgs_train5 = prepro_question(imgs_train5, params)
    # tokenization and preprocessing test question
    imgs_test5 = prepro_question(imgs_test5, params)

    # tokenization and preprocessing training paraphrase question
    imgs_train5 = prepro_question1(imgs_train5, params)
    # tokenization and preprocessing test paraphrase question
    imgs_test5 = prepro_question1(imgs_test5, params)

    
    # create the vocab for question
    imgs_train5,vocab = build_vocab_question(imgs_train5, params)


    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    
    ques_train5, ques_length_train5, question_id_train5 , cap_train5, cap_length_train5 = encode_question2(imgs_train5, params, wtoi)
    
    
    imgs_test5 = apply_vocab_question(imgs_test5, wtoi)
    
    
    ques_test5, ques_length_test5, question_id_test5 , cap_test5, cap_length_test5 = encode_question2(imgs_test5, params, wtoi)
    
    
   

    N = len(imgs_train5)
    f = h5py.File(params['output_h55'], "w")
    ## for train information
    f.create_dataset("ques_train", dtype='uint32', data=ques_train5)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train5)
    f.create_dataset("ques_cap_id_train", dtype='uint32', data=question_id_train5)#this is actually the ques_cap_id
    f.create_dataset("ques1_train", dtype='uint32', data=cap_train5)
    f.create_dataset("ques1_length_train", dtype='uint32', data=cap_length_train5)

    
    ## for test information
    f.create_dataset("ques_test", dtype='uint32', data=ques_test5)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test5)
    f.create_dataset("ques_cap_id_test", dtype='uint32', data=question_id_test5)
    f.create_dataset("ques1_test", dtype='uint32', data=cap_test5)
    f.create_dataset("ques1_length_test", dtype='uint32', data=cap_length_test5)

    f.close()
    print 'wrote ', params['output_h55']
    
    # create output json file
    
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    json.dump(out, open(params['output_json5'], 'w'))
    print 'wrote ', params['output_json5']
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input and output jsons and h5
    parser.add_argument('--input_train_json5', default='../data/quora_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json5', default='../data/quora_raw_test.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')
    parser.add_argument('--output_json5', default='../data/quora_data_prepro.json', help='output json file')
    parser.add_argument('--output_h55', default='../data/quora_data_prepro.h5', help='output h5 file')

   
    # options
    parser.add_argument('--max_length', default=26, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='tokenization method.')    
    parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
