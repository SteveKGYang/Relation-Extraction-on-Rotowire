import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable
import argparse
import h5py
from torch.nn.utils import clip_grad_norm
import time
import torch.nn.functional as F
import json
import os
import codecs
from text2num import text2num

class IEMODEL(nn.Module):
	def __init__(self, voc_size, emb_size, rnn_input, rnn_hidden, num_layers, rnn_bi, dp, rnn_mlp_size, label_num, enable_cnn, cnn_filter, cnn_kernel_width):
		super(IEMODEL, self).__init__()
		word_voc_size, entpos_voc_size, numpos_voc_size = voc_size
		word_emb_size, entpos_emb_size, numpos_emb_size = emb_size
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

		real_input_data = input_data.index_select(0, Variable(sorted_idx))
		return real_input_data, sorted_input_length, reverse_idx

	def reverse_sort_result(self, output, reverse_idx):
		# print(output.size())
		# print(torch.max(reverse_idx))
		return output.index_select(0, Variable(reverse_idx))

	# x are batch, length
	def forward(self, x, lengths):
		word, entpos, numpos = x

		word = self.word_emb(word)
		# print(entpos)
		# print(self.entpos_emb)
		entpos = self.entpos_emb(entpos)
		numpos = self.numpos_emb(numpos)
		assert word.size(1) == entpos.size(1) and word.size(1) == numpos.size(1)
		print(word.size())
		print(entpos.size())
		print(numpos.size())
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
parser.add_argument('-dataset', type=str, default=None,
					help="path to input")
parser.add_argument('-gen_text', type=str, default=None,
					help="path to input")
parser.add_argument('-gen_file', type=str, default=None,
					help="path to input")
parser.add_argument('-gen_inter_output', type=str, default=None,
					help="path to input")
parser.add_argument('-gen_json_path', type=str, default=None,
					help="path to input")
parser.add_argument('-gen_json_output', type=str, default=None,
					help="path to input")
parser.add_argument('-savepath', type=str, default=None,
					help="path to input")
parser.add_argument('-modelpath', type=str, default=None,
					help="path to input")
parser.add_argument('-mode', type=str, default=None,
					choices=["train", "test"],
					help="path to input")
parser.add_argument('-batch_size', type=int, default=None,
					help="path to input")
parser.add_argument('-word_emb_size', type=int, default=None,
					help="desired path to output file")
parser.add_argument('-rnn_hidden', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-rnn_mlp_hidden', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-epoch', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-dropout', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-lr', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-max_grad_norm', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-cnn', action="store_true",
					help="""Activate hier model 1""")
parser.add_argument('-cnn_filter', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-cnn_width', type=str, default=None,
					help="path to file containing generated summaries")

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
		prob.append(-torch.log((output[batch_no].index_select(0, label[batch_no][:label[batch_no][-1].data[0]])).sum()))

	return sum(prob) / output.size(0)

def get_acc(output, label, true_count, total_count):
	top_results = output.topk(1, dim=1)[1].squeeze(1)
	for batch_no in range(output.size(0)):
		total_count += 1
		if top_results.data[batch_no] in label[batch_no][:label[batch_no][-1].data[0]].data.cpu().numpy():
			true_count += 1

	return true_count, total_count

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
	# emb_sizes = (args.word_emb_size, args.word_emb_size // 2, args.word_emb_size // 2)
	emb_sizes = (768, args.word_emb_size // 2, args.word_emb_size // 2)
	# train
	param_init = 0.1

	word_dict = read_dict(args.dataset+".dict")
	labels_dict = read_dict(args.dataset+".labels")

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

	# vocab_sizes = (len(word_dict)+2, pos_voc_size+1, pos_voc_size+1)
	vocab_sizes = (28996, pos_voc_size+1, pos_voc_size+1)

	ie_model = IEMODEL(vocab_sizes, emb_sizes, sum(emb_sizes), args.rnn_hidden, 1, True, args.dropout, args.rnn_mlp_hidden, len(labels_dict) + 1, args.cnn, args.cnn_filter, list(map(int, args.cnn_width.strip().split(","))) if args.cnn_width is not None else None)

	for tmp_p in ie_model.parameters():
		tmp_p.data.uniform_(-param_init, param_init)

	print(ie_model)
	if args.gpu:
		ie_model.cuda()

	return ie_model	

def train(args, ie_model):
	# emb_sizes = (args.word_emb_size, args.word_emb_size // 2, args.word_emb_size // 2)
	# # train
	# param_init = 0.1

	# word_dict = read_dict(args.dataset+".dict")
	# labels_dict = read_dict(args.dataset+".labels")

	# h5fi = h5py.File(args.dataset+".h5", "r")
	# # iterate over epochs and batches
	# pos_voc_size = int(h5fi["max_dist"][0])
	# h5fi.close()

	# vocab_sizes = (len(word_dict), pos_voc_size+1, pos_voc_size+1)

	# ie_model = IEMODEL(vocab_sizes, emb_sizes, sum(emb_sizes), args.rnn_hidden, 1, True, args.dropout, args.rnn_mlp_hidden, len(labels_dict), args.cnn, args.cnn_filter, list(map(int, args.cnn_width.strip().split(","))) if args.cnn_width is not None else None)

	# for tmp_p in ie_model.parameters():
	# 	tmp_p.data.uniform_(-param_init, param_init)

	# print(ie_model)
	# ie_model.cuda()

	params = ie_model.parameters()
	optim = torch.optim.Adam(params, lr=args.lr)

	for epoch_no in range(args.epoch):
		print("training for epoch {}".format(epoch_no))
		start_time = time.time()
		h5fi = h5py.File(args.dataset+".h5", "r")
		# iterate over batches
		true_count = total_count = 0
		ie_model.train()
		for exp_no in range(0, len(h5fi["trsents"]), args.batch_size):
			words = Variable(torch.from_numpy(h5fi["trsents"][exp_no:exp_no+args.batch_size]).long())
			entpos = Variable(torch.from_numpy(h5fi["trentdists"][exp_no:exp_no+args.batch_size]).long()) + 300
			numpos = Variable(torch.from_numpy(h5fi["trnumdists"][exp_no:exp_no+args.batch_size]).long()) + 300
			labels = Variable(torch.from_numpy(h5fi["trlabels"][exp_no:exp_no+args.batch_size]).long())
			length = torch.from_numpy(h5fi["trlens"][exp_no:exp_no+args.batch_size]).long()
			if args.gpu:
				words = words.cuda()
				entpos = entpos.cuda()
				numpos = numpos.cuda()
				labels = labels.cuda()
				length = length.cuda()

			output = ie_model((words, entpos, numpos), length)
			loss = get_loss(output, labels)
			true_count, total_count = get_acc(output, labels, true_count, total_count)
			# training
			optim.zero_grad()
			loss.backward()
			clip_grad_norm(ie_model.parameters(), args.max_grad_norm)
			optim.step()

		train_prec = float(true_count) / total_count
		print("train prec for epoch {} is {}".format(epoch_no, float(true_count) / total_count))

		# validate after one epoch
		ie_model.eval()
		true_count = total_count = 0
		for exp_no in range(0, len(h5fi["valsents"]), args.batch_size):
			words = Variable(torch.from_numpy(h5fi["valsents"][exp_no:exp_no+args.batch_size]).long(), volatile=True)
			entpos = Variable(torch.from_numpy(h5fi["valentdists"][exp_no:exp_no+args.batch_size]).long(), volatile=True) + 300
			numpos = Variable(torch.from_numpy(h5fi["valnumdists"][exp_no:exp_no+args.batch_size]).long(), volatile=True) + 300
			labels = Variable(torch.from_numpy(h5fi["vallabels"][exp_no:exp_no+args.batch_size]).long(), volatile=True)
			length = torch.from_numpy(h5fi["vallens"][exp_no:exp_no+args.batch_size]).long()
			if args.gpu:
				words = words.cuda()
				entpos = entpos.cuda()
				numpos = numpos.cuda()
				labels = labels.cuda()
				length = length.cuda()

			output = ie_model((words, entpos, numpos), length)
			true_count, total_count = get_acc(output, labels, true_count, total_count)

		valid_prec = float(true_count) / total_count
		print("valid prec for epoch {} is {}".format(epoch_no, float(true_count) / total_count))
		print("time for one epoch is {}".format(time.time() - start_time))
		h5fi.close()

		torch.save(ie_model.state_dict(), args.savepath+"_train_{}_valid_{}.{}.pt".format("%.2f" % train_prec, "%.2f" % valid_prec, epoch_no))

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
	h5fi = h5py.File(args.gen_file, "r")
	true_count = total_count = 0
	results = []
	gen_texts = read_text(args.gen_text)
	for exp_no in range(0, len(h5fi["valsents"]), args.batch_size):
		words = Variable(torch.from_numpy(h5fi["valsents"][exp_no:exp_no+args.batch_size]).long(), volatile=True) + 1
		entpos = Variable(torch.from_numpy(h5fi["valentdists"][exp_no:exp_no+args.batch_size]).long(), volatile=True) + 94
		numpos = Variable(torch.from_numpy(h5fi["valnumdists"][exp_no:exp_no+args.batch_size]).long(), volatile=True) + 94
		labels = Variable(torch.from_numpy(h5fi["vallabels"][exp_no:exp_no+args.batch_size]).long(), volatile=True)
		length = torch.from_numpy(h5fi["vallens"][exp_no:exp_no+args.batch_size]).long().cuda()
		if args.gpu:
			words = words.cuda()
			entpos = entpos.cuda()
			numpos = numpos.cuda()
			labels = labels.cuda()
			length = length.cuda()

		output = ie_model((words, entpos, numpos), length)

		true_count, total_count = get_acc(output, labels, true_count, total_count)

		top_results = output.topk(1, dim=1)[1].squeeze(1)
		for batch_no in range(output.size(0)):
			results.append(top_results.data[batch_no])
		# true_count, total_count = get_acc(output, labels, true_count, total_count)

	assert len(results) == len(h5fi["valsents"])

	labels_dict = read_dict(args.dataset+".labels")
	rev_label_dict = {int(v):k for k, v in labels_dict.items()}
	boxrestartidxs = h5fi["boxrestartidxs"][:]
	'''
	with codecs.open(args.dataset+".maxdist", "r", "utf-8") as f:
		max_pos_dist, min_pos_dist = f.read().strip().split("\n")
		max_pos_dist = int(max_pos_dist)
		min_pos_dist = int(min_pos_dist)
	'''

	vocab_dict = read_dict(args.dataset+".dict")
	rev_vocab_dict = {int(v):k for k, v in vocab_dict.items()}

	final_results = []
	tmp_exps = []
	for exp_no, exp in enumerate(results):
		valentwords = []
		for idx, entdist in enumerate(h5fi["valentdists"][exp_no]):
			# if entdist + min_pos_dist == 0:
			if entdist == 0:
				valentwords.append(rev_vocab_dict[h5fi["valsents"][exp_no][idx]])
		valentwords = " ".join(valentwords)

		valnumwords = []
		for idx, numdist in enumerate(h5fi["valnumdists"][exp_no]):
			# if numdist + min_pos_dist == 0:
			if numdist == 0:
				valnumwords.append(rev_vocab_dict[h5fi["valsents"][exp_no][idx]])
		valnumwords = "".join(valnumwords)

		if exp_no != 0 and exp_no in boxrestartidxs:
			final_results.append(tmp_exps)
			tmp_exps = []
			for tmp_j in range(boxrestartidxs.tolist().count(exp_no)-1):
				final_results.append(tmp_exps)
			if rev_label_dict[results[exp_no]] not in ["PAD", "NONE"]:
				tmp_exps.append((valentwords, rev_label_dict[results[exp_no]] , valnumwords))

		elif rev_label_dict[results[exp_no]] not in ["PAD", "NONE"]:
			# find the ent and num
			tmp_exps.append((valentwords, rev_label_dict[results[exp_no]] , valnumwords))

	print("test prec is {}".format(float(true_count) / total_count))
	if len(tmp_exps) > 0:
		final_results.append(tmp_exps)
	# valid_prec = float(true_count) / total_count
	# print("valid prec for epoch {} is {}".format(epoch_no, float(true_count) / total_count))
	# print("time for one epoch is {}".format(time.time() - start_time))
	h5fi.close()

	# write a new json
	with open(args.gen_json_path, "r") as f:
		gen_json = json.load(f)
	# overwrite box/line score and summary
	PAD = "<blank>"
	assert len(gen_json) == len(final_results), (len(gen_json), len(final_results))
	assert len(gen_json) == len(gen_texts), (len(gen_json), len(gen_texts))

	for exp_no in range(len(gen_json)):
		special_ply_keys = ['FIRST_NAME', 'PLAYER_NAME', 'START_POSITION', 'SECOND_NAME']
		keep_ply_keys = ['TEAM_CITY']
		team_filter_keys = ['TEAM-PTS_QTR2', 'TEAM-FT_PCT', 'TEAM-PTS_QTR1', 'TEAM-PTS_QTR4', 'TEAM-PTS_QTR3', 'TEAM-PTS', 'TEAM-AST', 'TEAM-LOSSES', 'TEAM-WINS', 'TEAM-REB', 'TEAM-TOV', 'TEAM-FG3_PCT', 'TEAM-FG_PCT']
		mentioned_ply_idx = set([])
		id_ply_name = gen_json[exp_no]["box_score"]["PLAYER_NAME"]
		plyname_id = {v:k for k, v in id_ply_name.items()}
		box_related_info = {}
		home_realted_info = {}
		vis_related_info = {}
		home_match_name = gen_json[exp_no]["home_city"] + " " + gen_json[exp_no]["home_name"]
		vis_match_name = gen_json[exp_no]["vis_city"] + " " + gen_json[exp_no]["vis_name"]

		for each_triple in final_results[exp_no]:
			if "PLAYER-" in each_triple[1]:
				tmp_ply_key = each_triple[1].strip().split("-")[1]
				tmp_ply_id = []
				for each_name, each_idx in plyname_id.items():
					if each_triple[0] == each_name:
						tmp_ply_id.append(each_idx)
					
				if len(tmp_ply_id) <= 0:
					for each_name, each_idx in plyname_id.items():
						if each_triple[0] in each_name:
							tmp_ply_id.append(each_idx)

				tmp_ply_value = each_triple[2]
				if len(tmp_ply_id) > 1:
					print("warning: multiple ply matched")
				tmp_ply_id = tmp_ply_id[0] if len(tmp_ply_id) > 0 else "N/A"

				mentioned_ply_idx.add(tmp_ply_id)

				if tmp_ply_key not in box_related_info:
					box_related_info[tmp_ply_key] = {tmp_ply_id:tmp_ply_value}
				else:
					box_related_info[tmp_ply_key][tmp_ply_id] = tmp_ply_value

			elif "TEAM-" in each_triple[1]:
				tmp_team_id = None
				if each_triple[0] == vis_match_name:
					tmp_team_id = "vis"
				elif each_triple[0] == home_match_name:
					tmp_team_id = "home"
				elif each_triple[0] in vis_match_name:
					tmp_team_id = "vis"				
				else:
					tmp_team_id =  "home"


				tmp_team_key = each_triple[1].strip()
				tmp_team_value = each_triple[2].strip()

				if tmp_team_id == "home":
					home_realted_info[tmp_team_key] = tmp_team_value
				elif tmp_team_id == "vis":
					vis_related_info[tmp_team_key] = tmp_team_value

		if len(final_results[exp_no]) > 0:
			# change the box/line scores
			# change box score
			for k, v in gen_json[exp_no]["box_score"].items():
				if k not in special_ply_keys and k not in keep_ply_keys:
					new_values = {}
					for tmp_ply_idx, ply_score in gen_json[exp_no]["box_score"][k].items():
						if k in box_related_info and tmp_ply_idx in box_related_info[k]:
							new_values[tmp_ply_idx] = get_num(box_related_info[k][tmp_ply_idx])
						else:
							new_values[tmp_ply_idx] = PAD
					gen_json[exp_no]["box_score"][k] = new_values

				elif k in special_ply_keys:
					new_values = {}
					for tmp_ply_idx, ply_score in gen_json[exp_no]["box_score"][k].items():
						if tmp_ply_idx in mentioned_ply_idx:
							new_values[tmp_ply_idx] = ply_score
						else:
							new_values[tmp_ply_idx] = PAD
					gen_json[exp_no]["box_score"][k] = new_values

			# change line score
			new_line = {}
			for k, v in gen_json[exp_no]["home_line"].items():
				if k not in team_filter_keys:
					new_line[k] = v
				elif k in home_realted_info:
					new_line[k] = get_num(home_realted_info[k])
				else:
					new_line[k] = PAD
			gen_json[exp_no]["home_line"] = new_line

			new_line = {}
			for k, v in gen_json[exp_no]["vis_line"].items():
				if k not in team_filter_keys:
					new_line[k] = v
				elif k in vis_related_info:
					new_line[k] = get_num(vis_related_info[k])
				else:
					new_line[k] = PAD

			gen_json[exp_no]["vis_line"] = new_line

		# change summary
		gen_json[exp_no]["summary"] = gen_texts[exp_no]

	if args.gen_json_output is not None:
		makedir(args.gen_json_output)
		with open(args.gen_json_output, "w") as f:
			json.dump(gen_json, f)
	# write final_results into texts and create a new json
	write_inter_result = []
	
	for each_exp in final_results:
		write_inter_result.append(" ".join([",".join(tmp_exp) for tmp_exp in each_exp]))

	makedir(args.gen_inter_output)
	with open(args.gen_inter_output, "w") as f:
		f.write("\n".join(write_inter_result))

	# return final_results

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
		print("begin generating")
		ie_model.eval()
		test(args, ie_model)
# validate

# if is_training:
# 	self.optim.zero_grad()
# 	loss.backward()
# 	clip_grad_norm(self.seq2seq.parameters(),
#                    self.config["max_grad_norm"])
# 	self.optim.step()
