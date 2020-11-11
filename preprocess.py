import h5py
import json
import numpy as np
from transformers import DistilBertTokenizer

PAD = 0
CLS = 101
SEP = 102
NUM_MASK = '[NUM]'
ENT_MASK = '[ENT]'

# Load pretrained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./data2text/finetune_model')
tokenizer.add_tokens([NUM_MASK, ENT_MASK])
NUM_ID = tokenizer.encode(NUM_MASK, add_special_tokens=False)[0]
ENT_ID = tokenizer.encode(ENT_MASK, add_special_tokens=False)[0]

def get_entity_vocab():
    with open('sbnation/sbnation/train.json', 'r') as fr:
        train_data = json.load(fr)
    entity_vocab = set()
    for summ in train_data:
        team_entity = summ['home_name'].split() + summ['home_city'].split() + summ['vis_name'].split() + summ['vis_city'].split()
        player_entity = list(summ['box_score']['FIRST_NAME'].values()) + list(summ['box_score']['SECOND_NAME'].values())
        for ent in team_entity + player_entity:
            ent = ent.split('-')
            for e in ent:
                entity_vocab.add(e)
    with open('roto-ie.dict', 'r') as fr:
        roto_vocab = set([line.strip().split()[0] for line in fr.readlines()])
    with open('entity.dict', 'w') as fw:
        fw.write('\n'.join(list(entity_vocab & roto_vocab)))

def add_tokens():
    '''
    with open('roto-ie.dict', 'r') as fr:
        roto_vocab = [line.strip().split()[0] for line in fr.readlines()]
    with open('data/bert-base-cased-vocab.txt', 'r') as fr:
        bert_vocab = [line.strip() for line in fr.readlines()]

    tokenizer.add_tokens(list(set(roto_vocab) - set(bert_vocab)))
    '''
    with open('./sbnation/sbnation/entity.dict', 'r') as fr:
        ent_vocab = [line.strip().split()[0] for line in fr.readlines()]
    tokenizer.add_tokens(ent_vocab)

def transfer2bert():
    with open('roto-ie.dict', 'r') as fr:
        roto_vocab = {}
        for line in fr.readlines():
            w, i = line.strip().split()
            roto_vocab[int(i)] = w
    h5fi = h5py.File('roto-ie.h5', 'r')
    new_h5fi = h5py.File('./data2text/train.h5', 'w')
    for key in h5fi.keys():
        if 'label' in key:
            data = np.array(h5fi[key])
            new_h5fi[key] = data.copy()
    for data_type in ['tr', 'val', 'test']:
        sents = h5fi['%ssents'%data_type]
        numdists = h5fi['%snumdists'%data_type]
        entdists = h5fi['%sentdists'%data_type]
        new_sents = [] 
        new_nums = []
        new_ents = []
        new_lengths = []
        max_len = 0
        for sent, num_ds, ent_ds in zip(sents, numdists, entdists):
            new_sent_idxs = [CLS]
            new_num_ds = [num_ds[0] - 1]
            new_ent_ds = [ent_ds[0] - 1]  
            num_ids = []
            ent_ids = []
            num_pos = 0
            ent_pos = 0
            for idx, num_dist, ent_dist in zip(sent, num_ds, ent_ds):
                if idx != -1:
                    w = roto_vocab[idx]
                    new_w = tokenizer.encode(w, add_special_tokens=False)
                    if int(num_dist) == 0:
                        new_sent_idxs.extend([NUM_ID for _ in new_w])
                        num_ids += new_w
                        num_pos = ent_pos + 1
                    elif int(ent_dist) == 0:
                        new_sent_idxs.extend([ENT_ID for _ in new_w])
                        ent_ids += new_w
                        ent_pos = num_pos + 1
                    else:
                        new_sent_idxs.extend(new_w)
                    new_num_ds.extend([num_dist for _ in new_w])
                    new_ent_ds.extend([ent_dist for _ in new_w])
            if num_pos > ent_pos:
                new_sent_idxs.extend([SEP] + ent_ids + [SEP] + num_ids + [SEP])
            else:
                new_sent_idxs.extend([SEP] + num_ids + [SEP] + ent_ids + [SEP])
            # new_num.append(new_num[-1] + 1)
            # new_ent.append(new_ent[-1] + 1)
            if len(new_sent_idxs) > max_len:
                max_len = len(new_sent_idxs)
            new_sents.append(new_sent_idxs)
            new_nums.append(new_num_ds)
            new_ents.append(new_ent_ds)
            # assert len(new_sent_idxs) == len(new_num_ds) and len(new_num_ds) == len(new_ent_ds)
            new_lengths.append(len(new_sent_idxs))
        # add [PAD]
        for i in range(len(new_sents)):
            new_sents[i] += [0 for _ in range(max_len - len(new_sents[i]))]
            new_nums[i] += [new_nums[i][-1] + j + 1 for j in range(max_len - len(new_nums[i]))]
            new_ents[i] += [new_ents[i][-1] + j + 1 for j in range(max_len - len(new_ents[i]))]
        for i in range(len(new_sents) - 1):
            assert len(new_sents[i]) == len(new_sents[i + 1]) 
        new_h5fi["%ssents"%data_type] = np.array(new_sents, dtype=int)
        new_h5fi["%snumdists"%data_type] = np.array(new_nums, dtype=int)
        new_h5fi["%sentdists"%data_type] = np.array(new_ents, dtype=int)
        new_h5fi["%slens"%data_type] = np.array(new_lengths, dtype=int)
        print('%s is finished'%data_type)
    h5fi.close()
    new_h5fi.close()

if __name__ == '__main__':
    # get_entity_vocab()
    #add_tokens()
    transfer2bert()
