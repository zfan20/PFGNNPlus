# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
from collections import defaultdict
from tqdm import tqdm
import json
import argparse
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def write2file(edges, filename):
    with open(filename, 'w') as f:
        for eachedge in edges:
            f.write(str(eachedge[0]) + ' ' + str(eachedge[1]) + '\n')

def writeseq2file(seqs, filename):
    with open(filename, 'w') as f:
        for eachseq in seqs:
            f.write(str(eachseq[0]))
            for eachitem in eachseq[1:]:
                f.write(' ' + str(eachitem))
            f.write('\n')

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='Electronics', type=str)
parser.add_argument('--market', default='ae', type=str)
parser.add_argument('--rel_type', default='alsoBought', type=str)

args = parser.parse_args()


category_data_dir = '../raw_data/xmarketdata/' + args.data_name + '/'
amazon_review_meta_data_path = '../raw_data/meta_data/meta_' + args.data_name +'.json.gz'

meta_filepath = category_data_dir + 'meta_' + args.market + '_' + args.data_name + '.json.gz'

item_map_inamazon_review = set()
for eachitemdata in parse(amazon_review_meta_data_path):
    if 'title' not in eachitemdata or len(eachitemdata['title']) == 0:
        continue
    itemid = eachitemdata['asin']
    item_map_inamazon_review.add(itemid)


item_map = {}
itemind = 0
for eachitemdata in parse(meta_filepath):
    if 'title' not in eachitemdata or len(eachitemdata['title']) == 0 or args.rel_type not in eachitemdata['related'] or len(eachitemdata['related'][args.rel_type]) == 0:
        continue
    itemid = eachitemdata['asin']
    if itemid not in item_map_inamazon_review:
        continue
    if itemid not in item_map:
        item_map[itemid] = itemind
        itemind += 1


if not os.path.exists('./'+args.rel_type+'_edges/'):
    os.mkdir('./'+args.rel_type+'_edges/')

dataset_dir = './'+args.rel_type+'_edges/'+args.data_name+'_' + args.rel_type + '/'
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
dataset_market_dir = dataset_dir + args.market + '/'
if not os.path.exists(dataset_market_dir):
    os.mkdir(dataset_market_dir)
train_file = dataset_market_dir + 'train.edge'
test_file = dataset_market_dir + 'test.edge'
item_map_file = dataset_market_dir + 'item_map.pickle'
print('Dataset: ', args.data_name)
print('Market: ', args.market)
print('#of items: ', len(item_map))

all_edges = set()
for eachitemdata in parse(meta_filepath):
    if eachitemdata['asin'] not in item_map:
        continue

    itemind = item_map[eachitemdata['asin']]
    for eachitem in eachitemdata['related'][args.rel_type]:
        if eachitem not in item_map:
            continue
        eachitem_ind = item_map[eachitem]
        all_edges.add((itemind, eachitem_ind))
        all_edges.add((eachitem_ind, itemind))

print('#of total edges: ', len(all_edges))

edges_train, edges_test = train_test_split(list(all_edges), test_size=0.1, random_state=42)

print('#train edges: ', len(edges_train))
print('#test edges: ', len(edges_test))
write2file(edges_train, train_file)
write2file(edges_test, test_file)

with open(item_map_file, 'wb') as f:
    pickle.dump(item_map, f, protocol=pickle.HIGHEST_PROTOCOL)
