import h5py
import torch
import numpy as np
from pytorch_transformers import BertTokenizer
import time

PAD = 0
CLS = 101
SEP = 102

# Load pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('/users3/ywsun/project/data2text/data/bert-base-cased-vocab.txt')

def add_tokens():
    print(len(tokenizer))
    with open('roto-ie.dict', 'r') as fr:
        roto_vocab = [line.strip().split()[0] for line in fr.readlines()]
    with open('data/bert-base-cased-vocab.txt', 'r') as fr:
        bert_vocab = [line.strip() for line in fr.readlines()]
    with open('data/bert-base-cased-vocab.txt', 'r') as fr:
        bert_vocab_ = {}
        for i, line in enumerate(fr.readlines()):
            w = line.strip()
            bert_vocab_[i] = w
    '''
    cnt = 0
    for w in list(set(roto_vocab) - set(bert_vocab)):
        if len(tokenizer.encode(w, add_special_tokens=False)) != 1:
            print(w, [bert_vocab_[t] for t in tokenizer.encode(w, add_special_tokens=False)])
            cnt += 1
    print('unknown cnt %d'%cnt)
    '''
    tokenizer.add_tokens(list(set(roto_vocab) - set(bert_vocab)))
    print(len(tokenizer))

def transfer2bert():
    with open('roto-ie.dict', 'r') as fr:
        roto_vocab = {}
        for line in fr.readlines():
            w, i = line.strip().split()
            roto_vocab[int(i)] = w
    with open('data/bert-base-cased-vocab.txt', 'r') as fr:
        bert_vocab = {}
        for i, line in enumerate(fr.readlines()):
            w = line.strip()
            bert_vocab[i] = w
    fr = open('roto-ie.h5', 'rb')
    h5fi = h5py.File(fr, 'r')
    for data_type in ['tr', 'val', 'test']:
        start = time.time()
        print(h5fi['%ssents'%data_type].shape)
        sents = h5fi['%ssents'%data_type]
        numdists = h5fi['%snumdists'%data_type]
        entdists = h5fi['%sentdists'%data_type]
        new_sents = [] 
        new_nums = []
        new_ents = []
        new_lengths = []
        max_len = 0
        for sent, num, ent in zip(sents, numdists, entdists):
            new_sent_idxs = [CLS]
            new_num = [num[0] - 1]
            new_ent = [ent[0] - 1]
            ws = ' '.join([roto_vocab[w] for w in sent if w != -1])
            print(ws)  
            print(tokenizer.encode(ws, add_special_tokens=False))
            for idx, num_dist, ent_dist in zip(sent, num, ent):
                if idx != -1:
                    w = roto_vocab[idx]
                    new_w = tokenizer.encode(w, add_special_tokens=False)
                    new_sent_idxs.extend(new_w)
                    print(w, [bert_vocab[i] for i in new_w])
                    new_num.extend([num_dist for _ in new_w])
                    new_ent.extend([ent_dist for _ in new_w])
                new_num.append(new_num[-1] + 1)
                new_ent.append(new_ent[-1] + 1)
            new_sent_idxs.append(SEP)
            print(new_sent_idxs)
            exit()
            if len(new_sent_idxs) > max_len:
                max_len = len(new_sent_idxs)
            new_sents.append(new_sent_idxs)
            new_nums.append(new_num)
            new_ents.append(new_ent)
            assert len(new_sent_idxs) == len(new_num) and len(new_num) == len(new_ent)
            new_lengths.append(len(new_sent_idxs))
        # add [PAD]
        for i in range(len(new_sents)):
            new_sents[i] += [0 for _ in range(max_len - len(new_sents[i]))]
            new_nums[i] += [new_nums[i][-1] + j + 1 for j in range(max_len - len(new_nums[i]))]
            new_ents[i] += [new_ents[i][-1] + j + 1 for j in range(max_len - len(new_ents[i]))]
        print(time.time() - start)
        print('%s is finished'%data_type)
    h5fi.close()

if __name__ == '__main__':
    #w = "Belinelli"
    #print(tokenizer.encode(w, add_special_tokens=False))
    # add_tokens()
    #print(tokenizer.encode(w, add_special_tokens=False))
    transfer2bert()