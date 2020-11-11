import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np
import argparse
import h5py
from torch.nn.utils import clip_grad_norm_
import time
import torch.nn.functional as F
import json
import os
import codecs
from text2num import text2num

from transformers import DistilBertModel, DistilBertConfig,DistilBertTokenizer
from transformers import AdamW, WarmupLinearSchedule

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PAD = 0
NUM_MASK = '[NUM]'
ENT_MASK = '[ENT]'
SEP = '[SEP]'


class DistilBert(nn.Module):
	def __init__(self, load_pretrained_bert, bert_config):
		super(DistilBert, self).__init__()
		if(load_pretrained_bert):
			'''
			with open('roto-ie.dict', 'r') as fr:
				roto_vocab = [line.strip().split()[0] for line in fr.readlines()]
			with open('./data/bert-base-cased/vocab.txt', 'r') as fr:
				bert_vocab = [line.strip() for line in fr.readlines()]
			'''
			self.tokenizer = DistilBertTokenizer.from_pretrained('./ykl/_train_0.94_valid_0.94.3/bert/')
			#self.tokenizer.add_tokens(list(set(roto_vocab) - set(bert_vocab)))
			self.tokenizer.add_tokens([NUM_MASK, ENT_MASK])
			self.model = DistilBertModel.from_pretrained('./ykl/_train_0.94_valid_0.94.3/bert/')
			self.model.resize_token_embeddings(len(self.tokenizer))
		else:
			self.model = DistilBertModel(bert_config)

	def forward(self, words, attn_mask):
		words_output = self.model(words, attention_mask=attn_mask)
		return words_output

class IEMODEL(nn.Module):
	def __init__(self, voc_size, emb_size, rnn_input, rnn_hidden, num_layers, rnn_bi, dp, rnn_mlp_size, label_num, enable_cnn, cnn_filter, cnn_kernel_width, load_pretrained_bert, bert_config):
		super(IEMODEL, self).__init__()
		word_voc_size, entpos_voc_size, numpos_voc_size = voc_size
		word_emb_size, entpos_emb_size, numpos_emb_size = emb_size
		self.bert_model = DistilBert(load_pretrained_bert, bert_config)
		self.word_emb = nn.Embedding(word_voc_size, word_emb_size)
		# self.label_emb = nn.Embedding()
		self.entpos_emb = nn.Embedding(entpos_voc_size, entpos_emb_size)
		self.numpos_emb = nn.Embedding(numpos_voc_size, numpos_emb_size)

		self.rnn = nn.LSTM(rnn_input, rnn_hidden, num_layers=num_layers, dropout=dp, bidirectional=rnn_bi, batch_first=True)

		self.enable_cnn = enable_cnn
		if enable_cnn:
			Ci = 1
			Co = cnn_filter
			D = word_emb_size + entpos_emb_size + numpos_emb_size
			self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in cnn_kernel_width])
			self.cnn_mlp = nn.Sequential(
					nn.Linear(len(cnn_kernel_width) * Co, rnn_mlp_size),
					nn.ReLU(),
					nn.Dropout(dp)
				)		

		self.dropout = nn.Dropout(dp)

		self.rnn_mlp = nn.Sequential(
				nn.Linear(rnn_hidden*2, rnn_mlp_size),
				nn.ReLU(),
				nn.Dropout(dp)
			)

		if self.enable_cnn:
			decoder_input = rnn_mlp_size * 2
		else:
			decoder_input = rnn_mlp_size
		self.decode_layer = nn.Sequential(
				nn.Linear(decoder_input, label_num),
				nn.Softmax(1)
			)

	def sort_input(self, input_data, input_length):
		assert len(input_length.size()) == 1
		sorted_input_length, sorted_idx = torch.sort(input_length, dim=0, descending=True)

		_, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)

		real_input_data = input_data.index_select(0, sorted_idx)
		return real_input_data, sorted_input_length, reverse_idx

	def reverse_sort_result(self, output, reverse_idx):
		# print(output.size())
		# print(torch.max(reverse_idx))
		return output.index_select(0, reverse_idx)

	# x are batch, length
	def forward(self, x, lengths, attn_mask):
		word, entpos, numpos = x

		word = self.bert_model(word, attn_mask)
		# word = self.word_emb(word)
		# print(entpos)
		# print(self.entpos_emb)
		entpos = self.entpos_emb(entpos)
		numpos = self.numpos_emb(numpos)
		word = word[0]
		assert word.size(1) == entpos.size(1) and word.size(1) == numpos.size(1)
		# print(word.size())
		# print(entpos.size())
		# print(numpos.size())
		combined = torch.cat((word, entpos, numpos), 2)

		if self.enable_cnn:
			cnn_input = combined.unsqueeze(1)
			cnn_res = [F.relu(conv(cnn_input)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
			cnn_res = [F.max_pool1d(tmp_i, tmp_i.size(2)).squeeze(2) for tmp_i in cnn_res] # [(N, Co), ...]*len(Ks)
			cnn_res = torch.cat(cnn_res, 1)
			cnn_res = self.dropout(cnn_res)  # (N, len(Ks)*Co)
			cnn_res = self.cnn_mlp(cnn_res)  # (N, C)
		# rnn
		if lengths is not None:
			# Lengths data is wrapped inside a Variable.
			combined, lengths, reverse_idx = self.sort_input(combined, lengths)

			# entpos, ent_lengths, ent_reverse_idx = self.sort_input(entpos, lengths)
			# numpos, num_lengths, num_reverse_idx = self.sort_input(numpos, lengths)
			lengths = lengths.view(-1).tolist()
			packed_emb = pack(combined, lengths, batch_first=True)

		memory_bank, encoder_final = self.rnn(packed_emb)

		if lengths is not None:
			# memory_bank: batch, seqlen, hidden
			memory_bank = unpack(memory_bank, batch_first=True, padding_value=float("-Inf"))[0]
			memory_bank = self.reverse_sort_result(memory_bank, reverse_idx)

		# batch, hidden_size
		max_memory_bank = memory_bank.max(1)[0]
		mem_bank_mlp = self.rnn_mlp(max_memory_bank)
		if self.enable_cnn:
			mem_bank_mlp = torch.cat((mem_bank_mlp, cnn_res), 1)
		return self.decode_layer(mem_bank_mlp)

parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-gpu', action="store_true",
					help="""use cuda""")
parser.add_argument('-dataset', type=str, default='roto_cc-beam5_gens',
					help="path to input")
parser.add_argument('-gen_file', type=str, default='test',
					help="path to input")
parser.add_argument('-savepath', type=str, default=None,
					help="path to input")
parser.add_argument('-modelpath', type=str, default='./ykl/_train_0.94_valid_0.94.3/model.pt',
					help="path to input")
parser.add_argument('-mode', type=str, default=None,
					choices=["train", "test", "extract"],
					help="path to input")
parser.add_argument('-batch_size', type=int, default=1,
					help="path to input")
parser.add_argument('-word_emb_size', type=int, default=200,
					help="desired path to output file")
parser.add_argument('-rnn_hidden', type=int, default=768,
					help="path to file containing generated summaries")
parser.add_argument('-rnn_mlp_hidden', type=int, default=300,
					help="path to file containing generated summaries")
parser.add_argument('-epoch', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-dropout', type=float, default=0.2,
					help="prefix of .dict and .labels files")
parser.add_argument('-lr', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-max_grad_norm', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-cnn', action="store_true",
					help="""Activate hier model 1""")
parser.add_argument('-cnn_filter', type=int, default=200,
					help="path to file containing generated summaries")
parser.add_argument('-cnn_width', type=str, default='5',
					help="path to file containing generated summaries")
parser.add_argument('-load_pretrained_bert', action="store_true",
					help="whether to load the pretrained bert parameters")
parser.add_argument('-bert_config', type=str, default='./distil_bert/config.json',
					help="path to bert config file")

args = parser.parse_args()

def makedir(path):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        os.makedirs(path)

# output: batch, label_num
# label: batch, max_multiple_label
def get_loss(output, label):
	prob = []
	for batch_no in range(output.size(0)):
		# the last num in label list is the amount of labels
		prob.append(-torch.log((output[batch_no].index_select(0, label[batch_no][:label[batch_no][-1].data])).sum()))

	return sum(prob) / output.size(0)


def get_acc(output, label, true_count, total_count):
	top_results = output.topk(1, dim=1)[1].squeeze(1)
	for batch_no in range(output.size(0)):
		total_count += 1
		if top_results.data[batch_no] in label[batch_no][:label[batch_no][-1].data].data:
			true_count += 1
	return true_count, total_count


def get_multilabel_acc(output, label, true_count, total_count, nonenolabel, ignored):
	top_results = output.topk(1, dim=1)[1].squeeze(1)
	for batch_no in range(output.size(0)):
		if top_results.data[batch_no] == 11:
			ignored += 1
			continue
		total_count += 1
		if top_results.data[batch_no] in label[batch_no][:label[batch_no][-1].data].data:
			true_count += 1
	for batch_no in range(output.size(0)):
		if label[batch_no][0] != 11:
			nonenolabel += 1
	return true_count, total_count, nonenolabel, ignored


def read_text(path):
	texts = []
	with open(path, "r") as f:
		for line in f:
			if len(line.strip()) > 0:
				texts.append(line.strip().split(" "))
	return texts

def read_dict(path):
	dic = {}
	with open(path, "r") as f:
		for line in f:
			if len(line.strip()) > 0:
				word, idx = line.strip().split()
				dic[word] = idx
	return dic


def init_model(args):
	emb_sizes = (768, args.word_emb_size // 2, args.word_emb_size // 2)
	# train
	param_init = 0.1

	word_dict = read_dict(args.dataset + ".dict")
	labels_dict = read_dict(args.dataset + ".labels")
	# h5fi = h5py.File(args.dataset+".h5", "r")
	# # iterate over epochs and batches
	# pos_voc_size = int(h5fi["max_dist"][0])
	# h5fi.close()

	'''
	with open(args.dataset+".maxdist", "r") as f:
		pos_voc_size, _ = f.read().strip().split("\n")
		pos_voc_size = int(pos_voc_size)
	'''
	pos_voc_size = 600

	vocab_sizes = (len(word_dict)+2, pos_voc_size+1, pos_voc_size+1)

	ie_model = IEMODEL(vocab_sizes, emb_sizes, sum(emb_sizes), args.rnn_hidden, 1, True, args.dropout, args.rnn_mlp_hidden, len(labels_dict) + 1, args.cnn, args.cnn_filter, list(map(int, args.cnn_width.strip().split(","))) if args.cnn_width is not None else None, args.load_pretrained_bert, args.bert_config)

	print(ie_model)
	if args.gpu:
		ie_model.cuda()

	return ie_model	

def get_attention_mask(input_ids):
	attention_mask = torch.ones_like(input_ids)
	for i in range(input_ids.size(0)):
		for j in range(input_ids.size(1)):
			if int(input_ids[i, j]) == PAD:
				attention_mask[i, j] = 0
	return attention_mask

def train(args, ie_model):

	bert_params = list(map(id, ie_model.bert_model.parameters()))
	base_params = filter(lambda p: id(p) not in bert_params, ie_model.parameters())
	param_init = 0.1
	for tmp_p in base_params:
			tmp_p.data.uniform_(-param_init, param_init)
	#params = ie_model.parameters()
	#optim = torch.optim.Adam(params, lr=args.lr)
	optim = torch.optim.Adam([{'params': base_params}, {'params': ie_model.bert_model.parameters(), 'lr': 5e-5}], lr=args.lr)
	scheduler = WarmupLinearSchedule(optim, warmup_steps=8442, t_total=168840)
	h5fi = h5py.File(args.dataset+".h5", "r")

	for epoch_no in range(args.epoch):
		print("training for epoch {}".format(epoch_no))
		start_time = time.time()
		# iterate over batches
		true_count = total_count = nonenolabel = ignored = 0
		ie_model.train()
		for exp_no in range(0, len(h5fi["trsents"]), args.batch_size):
			words = torch.from_numpy(h5fi["trsents"][exp_no:exp_no+args.batch_size]).long()
			entpos = torch.from_numpy(h5fi["trentdists"][exp_no:exp_no+args.batch_size]).long() + 300
			numpos = torch.from_numpy(h5fi["trnumdists"][exp_no:exp_no+args.batch_size]).long() + 300
			labels = torch.from_numpy(h5fi["trlabels"][exp_no:exp_no+args.batch_size]).long()
			length = torch.from_numpy(h5fi["trlens"][exp_no:exp_no+args.batch_size]).long()
			attn_mask = get_attention_mask(words)
			if args.gpu:
				words = words.cuda()
				entpos = entpos.cuda()
				numpos = numpos.cuda()
				labels = labels.cuda()
				length = length.cuda()
				attn_mask = attn_mask.cuda()

			output = ie_model((words, entpos, numpos), length, attn_mask)
			loss = get_loss(output, labels)
			true_count, total_count, nonenolabel, ignored = get_multilabel_acc(output, labels, true_count, total_count, nonenolabel, ignored)
			# training
			optim.zero_grad()
			loss.backward()
			clip_grad_norm_(ie_model.parameters(), args.max_grad_norm)
			scheduler.step()
			optim.step()

		train_prec = float(true_count) / total_count
		print("train prec for epoch {} is {}, recall {}, ignored {}.".format(epoch_no,
			float(true_count) / total_count, float(true_count) / nonenolabel, float(ignored) / (ignored+total_count)))

		# validate after one epoch
		ie_model.eval()
		true_count = total_count = nonenolabel = ignored = 0
		for exp_no in range(0, len(h5fi["valsents"]), args.batch_size):
			words = torch.from_numpy(h5fi["valsents"][exp_no:exp_no+args.batch_size]).long()
			entpos = torch.from_numpy(h5fi["valentdists"][exp_no:exp_no+args.batch_size]).long() + 300
			numpos = torch.from_numpy(h5fi["valnumdists"][exp_no:exp_no+args.batch_size]).long() + 300
			labels = torch.from_numpy(h5fi["vallabels"][exp_no:exp_no+args.batch_size]).long()
			length = torch.from_numpy(h5fi["vallens"][exp_no:exp_no+args.batch_size]).long()
			attn_mask = get_attention_mask(words)
			if args.gpu:
				words = words.cuda()
				entpos = entpos.cuda()
				numpos = numpos.cuda()
				labels = labels.cuda()
				length = length.cuda()
				attn_mask = attn_mask.cuda()

			output = ie_model((words, entpos, numpos), length, attn_mask)
			true_count, total_count, nonenolabel, ignored = get_multilabel_acc(output, labels, true_count, total_count, nonenolabel, ignored)

		valid_prec = float(true_count) / total_count
		print("valid prec for epoch {} is {}, recall {}, ignored {}.".format(epoch_no, float(true_count) / total_count,
			  float(true_count) / nonenolabel, float(ignored) / (ignored + total_count)))
		print("time for one epoch is {}".format(time.time() - start_time))
		if not os.path.exists(args.savepath):
			os.makedirs(args.savepath)
		# ie_model.bert_model.save_pretrained('/users3/ywsun/project/data2text/models/')
		k = args.savepath+"/_train_{}_valid_{}.{}/".format("%.2f" % train_prec, "%.2f" % valid_prec, epoch_no)
		if not os.path.exists(k):
			os.makedirs(k)
		torch.save(ie_model.state_dict(), k+"model.pt")
		model_to_save = ie_model.bert_model.model
		if not os.path.exists(k+"bert"):
			os.makedirs(k+"bert")
		model_to_save.save_pretrained(k+"bert")
		ie_model.bert_model.tokenizer.save_pretrained(k+"bert")

	
	h5fi.close()

def get_num(strr):
	if strr.isdigit():
		return strr
	else:
		try:
			return str(text2num(strr))
		except:
			return "<blank>"
	# try:
	# 	return int(str)
	# except:
	# 	try:
	# 		return text2num(str)
	# 	except:
	# 		return "<blank>"

def test(args, ie_model):
	# validate after one epoch
	h5fi = h5py.File(args.dataset+".h5", "r")
	true_count = total_count = nonenolabel = ignored = 0
	#results = []
	#gen_texts = read_text(args.gen_text)
	for exp_no in range(0, len(h5fi["testsents"]), args.batch_size):
		words = torch.from_numpy(h5fi["testsents"][exp_no:exp_no+args.batch_size]).long()
		entpos = torch.from_numpy(h5fi["testentdists"][exp_no:exp_no+args.batch_size]).long() + 200
		numpos = torch.from_numpy(h5fi["testnumdists"][exp_no:exp_no+args.batch_size]).long() + 200
		labels = torch.from_numpy(h5fi["testlabels"][exp_no:exp_no+args.batch_size]).long()
		length = torch.from_numpy(h5fi["testlens"][exp_no:exp_no+args.batch_size]).long()
		attn_mask = get_attention_mask(words)

		if args.gpu:
			words = words.cuda()
			entpos = entpos.cuda()
			numpos = numpos.cuda()
			labels = labels.cuda()
			length = length.cuda()
			attn_mask = attn_mask.cuda()

		output = ie_model((words, entpos, numpos), length, attn_mask)

		true_count, total_count = get_acc(output, labels, true_count, total_count)
		#true_count, total_count, nonenolabel, ignored = get_multilabel_acc(output, labels, true_count, total_count, nonenolabel, ignored)

	print("test prec is {}".format(float(true_count) / total_count))
	#print("test prec is {}, recall {}, ignored {}.".format(float(true_count) / total_count,
					        #float(true_count) / nonenolabel, float(ignored) / (ignored + total_count)))


def extract(args, ie_model):
	dicts = []
	with open(args.dataset + '.dict', 'r') as f:
		for line in f:
			dicts.append(line.split()[0])
	relation_label = []
	with open(args.dataset+".labels", 'r') as f:
		for line in f:
			relation_label.append(line.split()[0])
	h5fi = h5py.File(args.dataset + ".h5", "r")

	tuple_file = open(args.gen_file+"-tuple.txt","w+")

	entpos = h5fi["valentdists"]
	numpos = h5fi["valnumdists"]
	# labels = torch.from_numpy(h5fi["vallabels"][exp_no:exp_no + args.batch_size]).long()
	leng = h5fi["vallens"]
	boxRestarts = h5fi["boxrestartidxs"]
	boxcount = 0
	labels = h5fi["vallabels"]
	origin_words = h5fi["valsents"]
	words = []
	length = []
	ent_pos = []
	num_pos = []
	ent = []
	num = []
	true_count = total_count = nonenolabel = ignored = zeros = 0
	for i in range(len(origin_words)):
		ents = []
		for j in range(len(entpos[i])):
			if entpos[i][j] == 0:
				ents.append(dicts[origin_words[i][j]-1])
		nums = []
		for j in range(len(numpos[i])):
			if numpos[i][j] == 0:
				nums.append(dicts[origin_words[i][j]-1])
		word = ['[CLS]']
		for j in range(leng[i]):
			wor = dicts[int(origin_words[i][j]) - 1]
			if wor in ents:
				word.append(ENT_MASK)
			elif wor in nums:
				word.append(NUM_MASK)
			else:
				word.append(wor)
		word.append(SEP)
		word += ents
		word.append(SEP)
		word += nums
		word.append(SEP)
		words.append(ie_model.bert_model.tokenizer.convert_tokens_to_ids(word))
		ent.append(ents)
		num.append(nums)

	for i in range(len(words)):
		length.append(len(words[i]))
	length = torch.tensor(length)
	max_len = 0
	for i in range(len(words)):
		if len(words[i]) > max_len:
			max_len = len(words[i])
	for i in range(len(words)):
		words[i] += [PAD]*(max_len-len(words[i]))
		b = [entpos[i][0]-1] + entpos[i].tolist()
		ent_len = len(b)
		if max_len <= ent_len:
			b = b[0:max_len]
		else:
			b += [j for j in range(b[-1] + 1, b[-1] + 1 + max_len - ent_len)]
		ent_pos.append(b)

		b = [numpos[i][0]-1] + numpos[i].tolist()
		num_len = len(b)
		if max_len <= num_len:
			b = b[0:max_len]
		else:
			b += [j for j in range(b[-1] + 1, b[-1] + 1 + max_len - num_len)]
		num_pos.append(b)
	words = torch.tensor(words)
	ent_pos = torch.tensor(ent_pos)
	num_pos = torch.tensor(num_pos)
	attn_mask = get_attention_mask(words)
	for i in range(words.size(0)):

		if args.gpu:
			wordss = words[i].unsqueeze(0).cuda()
			entposs = ent_pos[i].unsqueeze(0).cuda() + 300
			numposs = num_pos[i].unsqueeze(0).cuda() + 300
			lengths = length[i].unsqueeze(0).cuda()
			attn_masks = attn_mask[i].unsqueeze(0).cuda()
			label = torch.from_numpy(labels[i]).unsqueeze(0).cuda()
		else:
			wordss = words[i].unsqueeze(0)
			entposs = ent_pos[i].unsqueeze(0) + 250
			numposs = num_pos[i].unsqueeze(0) + 250
			lengths = length[i].unsqueeze(0)
			attn_masks = attn_mask[i].unsqueeze(0)
			label = torch.from_numpy(labels[i]).unsqueeze(0)

		output = ie_model((wordss, entposs, numposs), lengths, attn_masks)
		true_count, total_count, nonenolabel, ignored = get_multilabel_acc(output, label, true_count, total_count,
																		   nonenolabel, ignored)

		if boxcount < len(boxRestarts) and i + 1 == boxRestarts[boxcount]:
			tuple_file.write("\n")
			boxcount += 1

		#maxarg = output[0].argmax()
		if output[0].argmax() == 0:
			zeros += 1
			continue
		relation = relation_label[output[0].argmax()-1]
		if relation == 'NONE':
			continue
		e = ' '.join(ent[i])
		n = ' '.join(num[i])
		tuple_file.write(e+'|'+n+'|'+relation+'\n')
	print("test prec is {}, recall {}, ignored {}.".format(float(true_count) / total_count,
								float(true_count) / nonenolabel, float(ignored) / (ignored + total_count)))
	print("zeros {}".format(zeros))


	h5fi.close()
	tuple_file.close()



if __name__ == "__main__":
	if args.mode == "train":
		ie_model = init_model(args)
		makedir(args.savepath)
		# ie_model.train()
		train(args, ie_model)

	elif args.mode == "test":
		ie_model = init_model(args)
		# load state dict
		ie_model.load_state_dict(torch.load(args.modelpath))
		ie_model.eval()
		test(args, ie_model)
	elif args.mode == "extract":
		ie_model = init_model(args)
		# load state dict
		ie_model.load_state_dict(torch.load(args.modelpath))
		print("begin extracting")
		ie_model.eval()
		extract(args, ie_model)
