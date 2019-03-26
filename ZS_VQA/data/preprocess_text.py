#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import collections
import argparse
import string
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
import en_core_web_sm

try:
    import cPickle as pickle
except:
    import pickle

nlp = en_core_web_sm.load()
tokenizer = Tokenizer(nlp.vocab)
exclude = set(string.punctuation)


def process_answers(q, phase, n_answers=2000):

    # find the n_answers most common answers
    counts = {}
    for row in q:
        word = row['answer']
        counts[word] = counts.get(word, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)

    vocab = [w for c, w in cw[:n_answers]]
    file = open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/share_word.txt', 'r')
    share_word = [line.strip('\n') for line in file.readlines()]
    for word in share_word:
        if word not in vocab:
            vocab.append(word)

    # vocab = [w for c, w in cw]

    # a 0-indexed vocabulary translation table
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open(phase + '_a_dict.p', 'wb'))

    q = remove_examples(q, wtoi)
    q = encode_answer(q, wtoi)

    for row in q:
        accepted_answers = 0
        for w, c in row['answers']:
            if w in vocab:
                accepted_answers += c

        answers_scores = []
        for w, c in row['answers']:
            if w in vocab:
                answers_scores.append((w, c / accepted_answers))

        row['answers_w_scores'] = answers_scores

    return q


def remove_examples(examples, ans_to_aid):
    new_examples = []
    for ex in examples:
        if ex['answer'] in ans_to_aid:
            new_examples.append(ex)
    print('Number of examples reduced from %d to %d (%f%%)'%(len(examples), len(new_examples),
        (len(examples) - len(new_examples)) / float(len(examples)) * 100))
    return new_examples


def process_questions(q):
    # build question dictionary
    def build_vocab(questions):
        count_thr = 5
        # count up the number of times a word is used
        counts = {}
        for row in questions:
            for word in row['question_toked_UNK']:
                counts[word] = counts.get(word, 0) + 1
        cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
        print('top words and their counts:')
        print('\n'.join(map(str, cw[:10])))

        # print some stats
        total_words = sum(counts.values())
        print('total words:', total_words)
        bad_words = [w for w, n in counts.items() if n <= count_thr]
        vocab = [w for w, n in counts.items() if n > count_thr]
        bad_count = sum(counts[w] for w in bad_words)
        print('number of bad words: %d/%d = %.2f%%' %
              (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
        print('number of words in vocab would be %d' % (len(vocab), ))
        print('number of UNKs: %d/%d = %.2f%%' %
              (bad_count, total_words, bad_count*100.0/total_words))

        vocab.append(u'UNK')
        for ex in questions:
            words = ex['question_toked_UNK']
            question = [w if counts.get(w, 0) > count_thr else u'UNK' for w in words]
            ex['question_toked_UNK'] = question

        return questions, vocab

    q, vocab = build_vocab(q)
    # a 1-indexed vocab translation table

    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table
    pickle.dump({'itow': itow, 'wtoi': wtoi}, open('train_q_dict.p', 'wb'))

    q = encode_question(q, wtoi)
    json.dump(q, open('vqa_' + phase + '_final_3000.json', 'w'))


def tokenize_questions(qa, phase):
    qas = len(qa)
    for i, row in enumerate(tqdm(qa)):
        row['question_toked'] = [t.text if '?' not in t.text else t.text[:-1]
                                 for t in tokenizer(row['question'].lower())]  # get spacey tokens and remove question marks
        row['answer_toked'] = [t.text if ',' not in t.text else t.text[:-1]
                               for t in tokenizer(row['answer'].lower())]
        if i == qas - 1:
            json.dump(qa, open('vqa_' + phase + '_toked_new.json', 'w'))


def encode_question(examples, word_to_wid, maxlength=15, pad='right'):
    # Add to tuple question_wids and question_length
    for i, ex in enumerate(examples):
        ex['question_length'] = min(maxlength, len(ex['question_toked_UNK'])) # record the length of this sequence
        ex['question_wids'] = [0] * maxlength
        question_toked = ex['question_toked_UNK']


        for k, w in enumerate(question_toked):
            if k < maxlength:
                if pad == 'right':
                    ex['question_wids'][k] = word_to_wid[w]
                else:   #['pad'] == 'left'
                    new_k = k + maxlength - len(ex['question_toked_UNK'])
                    ex['question_wids'][new_k] = word_to_wid[w]

        ex['seq_length'] = len(ex['question_toked_UNK'])
    return examples


def encode_answer(examples, ans_to_aid):
    print('Warning: aid of answer not in vocab is -1')
    for i, ex in enumerate(examples):
        # for word in ex['answer_toked']:
        ex['answer_aid'] = [ans_to_aid.get(ex['answer'], -1)] # -1 means answer not in vocab
    return examples


def remove_long_tail_test(examples, word_to_wid):
    for ex in examples:
        ex['question_toked_UNK'] = [w if w in word_to_wid else 'UNK' for w in ex['question_toked']]
    return examples


def combine_qa(questions, annotations, phase):
    # Combine questions and answers in the same json file
    # 443757 questions
    data = []
    question_type = []

    with open('/mnt/data/xiaojinhui/wangtan/VQG/question_type_v2.txt', 'rb') as f:
        for line in f:
            question_type.append(line.decode().strip())

    for i, q in enumerate(tqdm(questions['questions'])):
        row = {}
        limited_answer = ['yes', 'no']
        # load questions info
        if annotations[i]['question_type'] in question_type and annotations[i]['multiple_choice_answer'] not in limited_answer:
            row['question'] = q['question']
            row['question_id'] = q['question_id']
            row['image_id'] = str(q['image_id'])

        # load answers
            assert q['question_id'] == annotations[i]['question_id']
            row['answer'] = annotations[i]['multiple_choice_answer']

            answers = []
            for ans in annotations[i]['answers']:
                answers.append(ans['answer'])
            row['answers'] = collections.Counter(answers).most_common()

            data.append(row)

        else:
            continue

    json.dump(data, open('vqa_' + phase + '_combined_new.json', 'w'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Preprocessing for VQA v2 text data')
    parser.add_argument('-d', '--data', nargs='+', help='train, val and/or test, list of data phases to be processed', required=True)
    parser.add_argument('--n', '--nanswers', default=2000, help='number of top answers to consider for classification.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))

    phase_list = args.data
    raw = '/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/'

    for phase in phase_list:

        if not os.path.exists(raw + 'zsa_trainingset.json'):
            raise SystemExit('Must download data first')

        print('processing ' + phase + ' data')

    # Build dictionary for question and answers (train set)
    if not os.path.exists('vqa_train_final_3000.json'):
        print('Building train dictionary...')
        t = json.load(open(raw + 'zsa_trainingset_.json'))
        tt = json.load(open(raw + 'vqa2_normal_train.json'))
        t = t + tt
        # if phase is 'train':
        phase = 'train'
        t = process_answers(t, phase, n_answers=args.n)
        process_questions(t)


        # Build dictionary for question and answers (val set)
    if not os.path.exists('vqa_val_final_3000.json'):
        answer_vocab = pickle.load(open('train_a_dict_vqa.p', 'rb'))
        ques_vocab = pickle.load(open('train_q_dict_vqa.p', 'rb'))

        t = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/data/vqa2_zsa_test.json'))
        t = remove_examples(t, answer_vocab['wtoi'])
        t = encode_answer(t, answer_vocab['wtoi'])
        t = remove_long_tail_test(t, ques_vocab['wtoi'])
        t = encode_question(t, ques_vocab['wtoi'])

        json.dump(t, open('vqa_val_final_3000.json', 'w'))

    if not os.path.exists('vqa_test_finalzsa_3000.json'):
        answer_vocab = pickle.load(open('train_a_dict_vqa.p', 'rb'))
        ques_vocab = pickle.load(open('train_q_dict_vqa.p', 'rb'))

        t = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_test_final_3000.json'))
        t = remove_examples(t, answer_vocab['wtoi'])
        t = encode_answer(t, answer_vocab['wtoi'])
        t = remove_long_tail_test(t, ques_vocab['wtoi'])
        t = encode_question(t, ques_vocab['wtoi'])

        json.dump(t, open('vqa_test_finalzsa_3000.json', 'w'))




    # Build test split
    # if not os.path.exists('vqa_test_final_3000.json'):
    #     val_all = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_val_final_3000.json'))
    #     split = json.load(open('/mnt/data/xiaojinhui/wangtan/VQG/dataset_coco.json'))
    #
    #     test_idx = []
    #     test_json = []
    #     new_image_id = []
    #     for i in split['images']:
    #         if i['split'] == 'test':
    #             test_idx.append(str(i['cocoid']))
    #
    #     for i in val_all:
    #         if i['image_id'] in test_idx:
    #             test_json.append(i)
    #             if i['image_id'] not in new_image_id:
    #                 new_image_id.append(i['image_id'])
    #
    #
    #     json.dump(test_json, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_test_final_3000.json', 'w'))
    #
    # if not os.path.exists('vqa_valk_final_3000.json'):
    #     val_all = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_val_final_3000.json'))
    #     split = json.load(open('/mnt/data/xiaojinhui/wangtan/VQG/dataset_coco.json'))
    #
    #     test_idx = []
    #     test_json = []
    #     new_image_id = []
    #     for i in split['images']:
    #         if i['split'] == 'val':
    #             test_idx.append(str(i['cocoid']))
    #
    #     for i in val_all:
    #         if i['image_id'] in test_idx:
    #             test_json.append(i)
    #             if i['image_id'] not in new_image_id:
    #                 new_image_id.append(i['image_id'])
    #
    #     json.dump(test_json, open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_valk_final_3000.json', 'w'))




    # For the zsa test, divide it into train or val

    # val_set = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/data/vqa_val_final_3000.json'))
    # val_idx = [pair['image_id'] for pair in val_set]
    # zsa_test = json.load(open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/vqa_val_final_3000.json'))
    #
    # test_split = []
    # train_split = []
    # for idx, pair in enumerate(zsa_test):
    #     if pair['image_id'] in val_idx:
    #         test_split.append(idx)
    #     else:
    #         train_split.append(idx)
    #
    #
    # trainfile = open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/train_split.txt', 'w')
    # for i in train_split:
    #     trainfile.write(i)
    #     trainfile.write('\n')
    # trainfile.close()
    #
    # testfile = open('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA/test_split.txt', 'w')
    # for i in test_split:
    #     testfile.write(i)
    #     testfile.write('\n')
    # testfile.close()
    #
    # print('Done')
