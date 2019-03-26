from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np
import h5py
import pandas as pd
from spacy.tokenizer import Tokenizer
import en_core_web_sm
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import utils
import torch
try:
    import cPickle as pickle
except:
    import pickle as pickle
nlp = en_core_web_sm.load()
tokenizer = Tokenizer(nlp.vocab)



class VQA_Dataset(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):

        # Set parameters
        self.data_dir = data_dir  # directory where the data is stored
        self.emb_dim = emb_dim    # answer embedding dimension
        self.train = train        # train (True) or eval (False) mode
        # self.seqlen = 14          # maximum question sequence length

        # Load training question dictionary
        q_dict = pickle.load(
            open(os.path.join(data_dir, 'train_q_dict_vqa.p'), 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        # Load training answer dictionary
        a_dict = pickle.load(
            open(os.path.join(data_dir, 'train_a_dict_vqa.p'), 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        # Load image features and bounding boxes
        # self.i_feat = zarr.open(os.path.join(
        #     data_dir, 'trainval.zarr'), mode='r')
        # self.bbox = zarr.open(os.path.join(
        #     data_dir, 'trainval_boxes.zarr'), mode='r')


        # Load questions
        if train:
            self.vqa = json.load(
                open(os.path.join('/mnt/data/xiaojinhui/wangtan_MM/vqa-project/ZS_VQA', 'vqa_train_final_3000.json')))
            self.i_feat = h5py.File(os.path.join(data_dir, 'train_feature.hdf5'), 'r', libver='latest')
            self.bbox = h5py.File(os.path.join(data_dir, 'train_box.hdf5'), 'r', libver='latest')
            self.cls = h5py.File(os.path.join(data_dir, 'train_cls.hdf5'), 'r', libver='latest')
            self.sizes = pd.read_csv(os.path.join(
                data_dir, 'train_image_size.csv'))

        else:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'vqa_valk_final_3000.json')))
            self.i_feat = h5py.File(os.path.join(data_dir, 'val_feature.hdf5'), 'r', libver='latest')
            self.bbox = h5py.File(os.path.join(data_dir, 'val_box.hdf5'), 'r', libver='latest')
            self.cls = h5py.File(os.path.join(data_dir, 'val_cls.hdf5'), 'r', libver='latest')
            self.sizes = pd.read_csv(os.path.join(
                data_dir, 'val_image_size.csv'))

        self.n_questions = len(self.vqa)

        print('Loading done')
        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        # self.embed_index = self.init_pretrained_wemb(emb_dim)
        self.embed_indexq = self.init_pretrained_wemb_q(emb_dim)

    def get_embeding(self, word, index):
        if word in index:
            emb = index.get(word)
        else:
            emb = torch.zeros(300)
        return emb


    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.n_answers, emb_dim), dtype=np.float32)
        for word, i in self.a_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v

        self.pretrained_wemb = embedding_mat

        return embeddings_index


    def init_pretrained_wemb_q(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.q_words, emb_dim), dtype=np.float32)
        for word, i in self.q_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v

        self.pretrained_wemb_q = embedding_mat

        return embeddings_index

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        q = self.vqa[idx]['question_wids']


        # q = [0] * self.seqlen
        # qlen = len(self.vqa[idx]['question_toked'])
        # # print(self.vqa[idx]['question_toked'])
        # for i, w in enumerate(self.vqa[idx]['question_toked']):
        #     try:
        #         q[i] = self.q_wtoi[w]
        #     except:
        #         q[i] = 0    # validation questions may contain unseen word

        # answer sample
        an = self.vqa[idx]['answer_aid']

        # id of the question
        # qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']
        img = np.asarray(self.i_feat[str(iid)])
        bboxes = np.asarray(self.bbox[str(iid)])

        # clses = self.cls[str(iid)].value[1:-1].split(', ')
        # cls_feat = np.zeros((36, 600))
        # for word_i, word in enumerate(clses):
        #     # print(word_i)
        #     clses_toked = tokenizer(word[1:-1].lower().decode())
        #     if len(clses_toked) > 2:
        #         clses_toked = clses_toked[0:2]
        #     for j, t in enumerate(clses_toked):
        #         word_em = self.get_embeding(str(t), self.embed_index)
        #         cls_feat[word_i, j*300:(j+1)*300] = word_em
        # an_feat = np.zeros((len(an), 300))
        # for i, word_idx in enumerate(an):
        #     an_feat[i] = self.pretrained_wemb[word_idx]


        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # number of image objects
        k = 36

        # scale bounding boxes by image dimensions
        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox

        # format variables
        q = np.asarray(q)
        q = torch.from_numpy(q)
        # a = np.asarray(a).reshape(-1)
        an = np.asarray(an)
        an = torch.from_numpy(an)
        # n_votes = np.asarray(n_votes).reshape(-1)
        # qid = np.asarray(qid).reshape(-1)
        # i = np.concatenate([img, bboxes], axis=1)
        i = torch.from_numpy(img)
        target = torch.zeros(self.n_answers)
        # k = np.asarray(k).reshape(1)

        # return q, a, n_votes, qid, i, k, qlen, an, clses
        # q = torch.from_numpy(q)
        # an_feat = torch.from_numpy(an_feat).float()
        # cls_feat = torch.from_numpy(cls_feat).float()
        # img_feat = torch.from_numpy(i).float()
        # area_idx = utils.find_area(an_feat, cls_feat).view(-1)
        # an_feat = torch.mean(an_feat, 0)

        # adj_mat = torch.eye(36)
        # adj_mat[area_idx] = 1.
        # adj_mat[:, area_idx] = 1.

        return i, q, an



class VQA_Dataset_Test(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):

        # Set parameters
        self.data_dir = data_dir  # directory where the data is stored
        self.emb_dim = emb_dim    # answer embedding dimension
        self.train = train        # train (True) or eval (False) mode
        # self.seqlen = 14          # maximum question sequence length

        # Load training question dictionary
        q_dict = pickle.load(
            open(os.path.join(data_dir, 'train_q_dict_vqa.p'), 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        # Load training answer dictionary
        a_dict = pickle.load(
            open(os.path.join(data_dir, 'train_a_dict_vqa.p'), 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        # Load questions

        self.vqa = json.load(
            open('vqa_val_final_3000.json'))
        self.i_feat = h5py.File(os.path.join(data_dir, 'train_feature.hdf5'), 'r', libver='latest')
        self.i_feat_val = h5py.File(os.path.join(data_dir, 'val_feature.hdf5'), 'r', libver='latest')
        self.bbox = h5py.File(os.path.join(data_dir, 'train_box.hdf5'), 'r', libver='latest')
        self.bbox_val = h5py.File(os.path.join(data_dir, 'val_box.hdf5'), 'r', libver='latest')
        self.cls = h5py.File(os.path.join(data_dir, 'train_cls.hdf5'), 'r', libver='latest')
        self.cls_val = h5py.File(os.path.join(data_dir, 'val_cls.hdf5'), 'r', libver='latest')
        self.sizes = pd.read_csv(os.path.join(data_dir, 'train_image_size.csv'))
        self.sizes_val = pd.read_csv(os.path.join(data_dir, 'val_image_size.csv'))

        self.n_questions = len(self.vqa)

        print('Loading done')
        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        self.embed_index = self.init_pretrained_wemb(emb_dim)

    def get_embeding(self, word, index):
        if word in index:
            emb = index.get(word)
        else:
            emb = torch.zeros(300)
        return emb


    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.n_answers, emb_dim), dtype=np.float32)
        for word, i in self.a_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v

        self.pretrained_wemb = embedding_mat

        return embeddings_index


    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        q = self.vqa[idx]['question_wids']

        # answer sample
        an = self.vqa[idx]['answer_aid']


        # id of the question
        qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']
        if idx < 11378:  # 0~11377
            img = np.asarray(self.i_feat[str(iid)])
            bboxes = np.asarray(self.bbox[str(iid)])
            # clses = self.cls[str(iid)].value[1:-1].split(', ')
            imsize = self.sizes[str(iid)]
        else:
            img = np.asarray(self.i_feat_val[str(iid)])
            bboxes = np.asarray(self.bbox_val[str(iid)])
            # clses = self.cls_val[str(iid)].value[1:-1].split(', ')
            imsize = self.sizes_val[str(iid)]

        # number of votes for each answer
        n_votes = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.vqa[idx]['answers']:
            try:
                n_votes[self.a_wtoi[w]] = c
            except:
                continue



        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # number of image objects
        k = 36

        # scale bounding boxes by image dimensions
        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox

        # format variables
        q = np.asarray(q)
        q = torch.from_numpy(q)
        # a = np.asarray(a).reshape(-1)
        i = torch.from_numpy(img)
        an = np.asarray(an)
        an = torch.from_numpy(an)
        n_votes = np.asarray(n_votes).reshape(-1)


        return i, q, an, n_votes


class VQA_Dataset_eva(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):

        # Set parameters
        self.data_dir = data_dir  # directory where the data is stored
        self.emb_dim = emb_dim    # answer embedding dimension
        self.train = train        # train (True) or eval (False) mode
        # self.seqlen = 14          # maximum question sequence length

        # Load training question dictionary
        q_dict = pickle.load(
            open(os.path.join(data_dir, 'train_q_dict_vqa.p'), 'rb'))
        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        # Load training answer dictionary
        a_dict = pickle.load(
            open(os.path.join(data_dir, 'train_a_dict_vqa.p'), 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        # Load questions

        self.vqa = json.load(
            open('vqa_test_finalzsa_3000.json'))
        self.i_feat_val = h5py.File(os.path.join(data_dir, 'val_feature.hdf5'), 'r', libver='latest')
        self.bbox_val = h5py.File(os.path.join(data_dir, 'val_box.hdf5'), 'r', libver='latest')
        self.cls_val = h5py.File(os.path.join(data_dir, 'val_cls.hdf5'), 'r', libver='latest')
        self.sizes_val = pd.read_csv(os.path.join(data_dir, 'val_image_size.csv'))

        self.n_questions = len(self.vqa)

        print('Loading done')
        self.feat_dim = self.i_feat_val[list(self.i_feat_val.keys())[
            0]].shape[1] + 4  # + bbox
        self.embed_index = self.init_pretrained_wemb(emb_dim)

    def get_embeding(self, word, index):
        if word in index:
            emb = index.get(word)
        else:
            emb = torch.zeros(300)
        return emb


    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()

        embedding_mat = np.zeros((self.n_answers, emb_dim), dtype=np.float32)
        for word, i in self.a_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat[i] = embedding_v

        self.pretrained_wemb = embedding_mat

        return embeddings_index


    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        q = self.vqa[idx]['question_wids']

        # answer sample
        an = self.vqa[idx]['answer_aid']


        # id of the question
        qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']

        img = np.asarray(self.i_feat_val[str(iid)])
        bboxes = np.asarray(self.bbox_val[str(iid)])
        # clses = self.cls_val[str(iid)].value[1:-1].split(', ')
        imsize = self.sizes_val[str(iid)]

        # number of votes for each answer
        n_votes = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.vqa[idx]['answers']:
            try:
                n_votes[self.a_wtoi[w]] = c
            except:
                continue



        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # number of image objects
        k = 36

        # scale bounding boxes by image dimensions
        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox

        # format variables
        q = np.asarray(q)
        q = torch.from_numpy(q)
        i = torch.from_numpy(img)
        # a = np.asarray(a).reshape(-1)
        an = np.asarray(an)
        an = torch.from_numpy(an)
        n_votes = np.asarray(n_votes).reshape(-1)


        return i, q, an, n_votes
