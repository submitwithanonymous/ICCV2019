import os
import json
import spacy
nlp = spacy.load("en")
import nltk
from nltk.corpus import stopwords
import random
random.seed(1000)

try:
    import cPickle as pickle
except:
    import pickle as pickle

question_vocab = pickle.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/train_q_dict.p', 'rb'))
answer_vocab = pickle.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/train_a_dict.p', 'rb'))
train_all = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_train_final_3000.json'))
val_all = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_val_final_3000.json'))


def construct_ques(share_word, trainset):
    question_set = {}
    for word in share_word:
        idx_list = []
        for idx, pair in enumerate(trainset):
            if word in pair['question_toked']:
                idx_list.append(idx)
        question_set[word] = idx_list

    return question_set


def mix_dataset(train, test):
    return train + test


def find_word(question_vocab, answer_vocab):
    share_word = []
    for question_word in question_vocab['wtoi']:
        if question_word in answer_vocab['wtoi'].keys():
            share_word.append(question_word)

    return share_word


def delete_stopwords(share_word):
    share_word = [word.lower() for word in share_word]
    share_word_stop = [word for word in share_word if word not in stopwords.words('english')]

    return share_word_stop


def list2txt(list, path):
    file = open(path, 'w')
    for i in list:
        file.write(i)
        file.write('\n')
    file.close()


def ZSA(vqaset, trainset, share_word):
    '''
    In training set the answer cannot include share_word
    :param share_word: list
    :return: question-answer pairs idx
    '''
    testing_idx = {}
    for i,pair in enumerate(vqaset):

        if pair['answer'] in share_word:
            testing_idx[pair['question_id']] = pair['answer']

    new_trainset = [pair for pair in trainset if pair['question_id'] not in testing_idx.keys()]
    new_testset = [pair for pair in vqaset if pair['question_id'] in testing_idx.keys()]

    return new_trainset, new_testset


def examine(share_word, train_json):
    '''
    Ensure that all the zero-shot answer have occurred in questions
    :param share_word: txt
    :param train_json: json
    :return: Yes or no
    '''
    file = open(share_word, 'r')
    zsa = [line.strip('\n') for line in file.readlines()]
    for word in zsa:
        flag = 0
        for pair in train_json:
            if word in pair['question_toked']:
                flag = 1
                continue
            else:
                continue
        if flag != 1:
            print word




if __name__ == '__main__':
    vqa_all = mix_dataset(train_all, val_all)
    share_word = find_word(question_vocab, answer_vocab)
    share_word_stop = random.sample(delete_stopwords(share_word), 137)
    noraml_train, zsa_test = ZSA(vqa_all, train_all, share_word_stop)
    json.dump(noraml_train, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_normal_train.json', 'w'))
    json.dump(zsa_test, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_zsa_test.json', 'w'))
    list2txt(share_word_stop, '/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/share_word.txt')

    # examine
    # share_word = '/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/share_word.txt'
    # train_json = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_normal_train.json'))
    # examine(share_word, train_json)

    # add the supplementary qustion-answer pair
    train_set = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_normal_train.json'))
    supp_trainset = []
    share_word = '/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/share_word.txt'
    file = open(share_word, 'r')
    share_word = [line.strip('\n') for line in file.readlines()]
    idx_list = construct_ques(share_word, train_set)
    for word in idx_list.keys():
        for idx in idx_list[word]:
            new_pair = train_set[idx]
            new_pair['answer'] = word
            new_pair['answer_toked'] = [word]
            new_pair['answer_aid'] = [answer_vocab['wtoi'].get(word, -1)]
            supp_trainset.append(new_pair)
    json.dump(supp_trainset, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_zsa_trainsupp.json', 'w'))



    print('Done!')