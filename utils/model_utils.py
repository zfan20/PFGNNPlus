# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import random
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from collections import defaultdict
import pickle

def neg_sample(spmat, itemind, n=1):
    pos_inds = set(spmat[itemind].nonzero()[1])
    while True:
        neg = np.random.randint(0, spmat.shape[1])
        if neg not in pos_inds:
            return neg

def get_mostfree_gpu():
    os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def read_item2item_xmarketdata_fromreviewedges(category_name, market, edge_type='alsoViewed_alsoBought'):
    review_edges_filepath = '../xmarket_preprocess/co_reviewed_edges4rel/' + category_name + '_' + edge_type + '/' + market + '/all.edge'
    test_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/test.edge'
    item_map_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/item_map.pickle'
    with open(item_map_filepath, 'rb') as f:
        item_map = pickle.load(f)
    max_itemind = max([iind for iid, iind in item_map.items()])

    train_dict = defaultdict(set)
    with open(review_edges_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            train_dict[item1].add(item2)
            train_dict[item2].add(item1)
    test_dict = defaultdict(list)
    with open(test_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            test_dict[item1].append(item2)
            test_dict[item2].append(item1)

    print("#of items: ", max_itemind+1)
    print("#of review edges: ", sum([len(ilist) for u, ilist in train_dict.items()]))

    return train_dict, test_dict, max_itemind+1



def read_item2item_xmarketdata_seq(category_name, market, edge_type='co_reviewed'):
    #../xmarket_preprocess/co_reviewed_edges/Electronics_co_reviewed/ae/
    train_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/train.edge'
    test_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/test.edge'
    trainseq_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/train_seq.txt'
    item_map_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/item_map.pickle'
    item_feature_filepath = '../data/meta_titledesc_features/' + category_name + '_itemtitledescriptiontextfeats.pickle'
    with open(item_map_filepath, 'rb') as f:
        item_map = pickle.load(f)
    item_ind2id_map = {ind: iid for iid, ind in item_map.items()}
    with open(item_feature_filepath, 'rb') as f:
        item_feature_dict = pickle.load(f)

    max_itemind = 0
    train_i, train_j = [], []
    train_valus = []
    with open(train_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            train_i.append(item1+1)
            train_j.append(item2+1)
            train_valus.append(1)

    train_seqs = []
    with open(trainseq_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            oneseq = []
            for item in parts:
                oneseq.append(int(item)+1)
            if len(oneseq) > 0:
                train_seqs.append(oneseq)

    test_dict = defaultdict(list)
    with open(test_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            test_dict[item1+1].append(item2+1)

    train_spmat = csr_matrix((train_valus, (train_i, train_j)), shape=(max_itemind+2, max_itemind+2))

    item_features = [0.02*np.random.randn(1, 768).astype(dtype=np.float32)[0]]
    for itemind in range(max_itemind+1):
        itemid = item_ind2id_map[itemind]
        feats = item_feature_dict[itemid]
        item_features.append(feats)
    item_features_x = torch.tensor(np.array(item_features), dtype=torch.float)

    return item_features_x, train_spmat, train_seqs, test_dict, max_itemind+1


#this function is used for co_reviewed data for xmarket, archieved
def read_item2item_xmarketdata(category_name, market, edge_type='co_reviewed'):
    #../xmarket_preprocess/co_reviewed_edges/Electronics_co_reviewed/ae/
    train_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/train.edge'
    test_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/test.edge'
    trainseq_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/train_seq.txt'
    item_map_filepath = '../xmarket_preprocess/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/' + market + '/item_map.pickle'
    item_feature_filepath = '../data/meta_titledesc_features/' + category_name + '_itemtitledescriptiontextfeats.pickle'
    with open(item_map_filepath, 'rb') as f:
        item_map = pickle.load(f)
    item_ind2id_map = {ind: iid for iid, ind in item_map.items()}
    with open(item_feature_filepath, 'rb') as f:
        item_feature_dict = pickle.load(f)

    max_itemind = 0
    train_i, train_j = [], []
    train_valus = []
    train_dict = defaultdict(list)
    with open(train_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            train_i.append(item1)
            train_j.append(item2)
            train_i.append(item2)
            train_j.append(item1)
            train_valus.append(1)
            train_valus.append(1)
            train_dict[item1].append(item2)
            train_dict[item2].append(item1)

    if edge_type == 'co_reviewed':
        train_seq = []
        with open(trainseq_filepath, 'r') as f:
            for eachline in f:
                parts = eachline.rstrip().split(' ')
                oneseq = []
                for item in parts:
                    oneseq.append(int(item)+1)
                if len(oneseq) > 0:
                    train_seq.append(oneseq)

    test_dict = defaultdict(list)
    with open(test_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            test_dict[item1].append(item2)
            test_dict[item2].append(item1)

    train_spmat = csr_matrix((train_valus, (train_i, train_j)), shape=(max_itemind + 1, max_itemind + 1))

    train_edge_index = torch.tensor([train_i, train_j], dtype=torch.long)
    item_features = []
    for itemind in range(max_itemind+1):
        itemid = item_ind2id_map[itemind]
        feats = item_feature_dict[itemid]
        item_features.append(feats)
    train_x = torch.tensor(np.array(item_features), dtype=torch.float)
    train_data = Data(x=train_x, edge_index=train_edge_index)

    print('Category: ', category_name)
    print('Market: ', market)
    print("#of items: ", max_itemind+1)
    print("#of train edges: ", len(train_i))
    if edge_type == 'co_reviewed':
        print("#of train seqs with length > 1: ", len(train_seq))
    print("#of test items: ", len(test_dict))
    
    return train_data, train_spmat, train_dict, test_dict, max_itemind+1


def read_item2item_data_fromreviewedges(category_name, edge_type='also_buy'):
    review_edges_filepath = '../preprocess_data/co_reviewed_edges4rel/' + category_name + '_' + edge_type + '/all.edge'
    test_filepath = '../preprocess_data/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/test.edge'
    item_map_filepath = '../preprocess_data/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/item_map.pickle'
    with open(item_map_filepath, 'rb') as f:
        item_map = pickle.load(f)
    max_itemind = max([iind for iid, iind in item_map.items()])

    train_dict = defaultdict(set)
    with open(review_edges_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            train_dict[item1].add(item2)
            train_dict[item2].add(item1)
    test_dict = defaultdict(list)
    with open(test_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            test_dict[item1].append(item2)
            test_dict[item2].append(item1)
    print("#of items: ", max_itemind+1)
    print("#of review edges: ", sum([len(ilist) for u, ilist in train_dict.items()]))
    return train_dict, test_dict, max_itemind+1


def read_item2item_data(category_name, edge_type='also_buy'):
    train_filepath = '../preprocess_data/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/train.edge'
    test_filepath = '../preprocess_data/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/test.edge'
    item_map_filepath = '../preprocess_data/' + edge_type + '_edges/' + category_name + '_' + edge_type + '/item_map.pickle'
    item_feature_filepath = '../data/meta_titledesc_features/' + category_name + '_itemtitledescriptiontextfeats.pickle'

    with open(item_map_filepath, 'rb') as f:
        item_map = pickle.load(f)
    item_ind2id_map = {ind: iid for iid, ind in item_map.items()}
    with open(item_feature_filepath, 'rb') as f:
        item_feature_dict = pickle.load(f)

    max_itemind = 0
    train_i, train_j = [], []
    train_valus = []
    train_dict = defaultdict(list)
    with open(train_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            # not directional edges
            train_i.append(item1)
            train_i.append(item2)
            train_j.append(item2)
            train_j.append(item1)
            train_valus.append(1)
            train_valus.append(1)
            train_dict[item1].append(item2)
            train_dict[item2].append(item1)
    test_dict = defaultdict(list)
    with open(test_filepath, 'r') as f:
        for eachline in f:
            parts = eachline.rstrip().split(' ')
            item1, item2 = int(parts[0]), int(parts[1])
            max_itemind = max(max_itemind, max(item1, item2))
            test_dict[item1].append(item2)
            test_dict[item2].append(item1)

    train_spmat = csr_matrix((train_valus, (train_i, train_j)), shape=(max_itemind + 1, max_itemind + 1))

    train_edge_index = torch.tensor([train_i, train_j], dtype=torch.long)
    item_features = []
    for itemind in range(max_itemind+1):
        itemid = item_ind2id_map[itemind]
        feats = item_feature_dict[itemid]
        #item_feature_dict: {asin:numpy.array(dim=768)}
        item_features.append(feats)
    train_x = torch.tensor(np.array(item_features), dtype=torch.float)
    train_data = Data(x=train_x, edge_index=train_edge_index)

    print('Category: ', category_name)
    print("#of items: ", max_itemind+1)
    print("#of train edges: ", len(train_i))
    print("#of test items: ", len(test_dict))

    return train_data, train_spmat, train_dict, test_dict, max_itemind+1


