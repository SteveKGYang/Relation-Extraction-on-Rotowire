import torch
import os
import time
import h5py
import numpy as np
import argparse

from torch.nn.utils import clip_grad_norm

from transformers import AdamW, WarmupLinearSchedule
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForMaskedLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PAD = 0
CLS = 101
SEP = 102
MASK = -1

parser = argparse.ArgumentParser(description='Utility Functions')

parser.add_argument('-gpu', action="store_true",
					help="""use cuda""")
parser.add_argument('-dataset', type=str, default=None,
					help="path to input")
parser.add_argument('-save_path', type=str, default=None,
					help="path to input")
parser.add_argument('-mode', type=str, default=None,
					choices=["train", "test"],
					help="path to input")
parser.add_argument('-batch_size', type=int, default=None,
					help="path to input")
parser.add_argument('-epoch', type=int, default=None,
					help="path to file containing generated summaries")
parser.add_argument('-lr', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument('-max_grad_norm', type=float, default=None,
					help="prefix of .dict and .labels files")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
args = parser.parse_args()


def train(args, model, tokenizer):
	
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	optim = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
	scheduler = WarmupLinearSchedule(optim, warmup_steps=8442, t_total=168840)
	f = open(args.dataset, "r")
	h5fi = [line.strip() for line in f]
	with open("./sbnation/sbnation/entity.dict", "r") as fl:
		entity_dict = [word.strip() for word in fl]

	tr_loss = 0.0
	global_step = 0

	for epoch_no in range(args.epoch):
		print("training for epoch {}".format(epoch_no))
		start_time = time.time()
		model.train()
		for exp_no in range(0, len(h5fi), args.batch_size):
			inputs = []
			words = h5fi[exp_no:exp_no+args.batch_size]
			am_labels = []
			ml_labels = []
			for i in range(args.batch_size):
				items = []
				if i == len(words):
					break
				lines = words[i].split(" ")
				if len(lines) > 512:
					continue
				nums = [i for i, a in enumerate(lines) if a.isdigit() or a in entity_dict]
				input_line = tokenizer.convert_tokens_to_ids(lines)
				am = [1]*len(input_line)
				ml = input_line.copy()
				for j in range(len(ml)):
					if j not in nums:
						ml[j] = -1
				inputs.append(input_line)
				am_labels.append(am)
				ml_labels.append(ml)

			max_len = 512
			'''for l in inputs:
				if len(l) > max_len:
					max_len = len(l)'''

			for i in range(len(inputs)):
				ol = len(inputs[i])
				for _ in range(max_len-ol):
					inputs[i].append(0)

				ol = len(am_labels[i])
				for _ in range(max_len-ol):
					am_labels[i].append(0)
				ol = len(ml_labels[i])
				for _ in range(max_len-ol):
					ml_labels[i].append(0)

			inputs = torch.from_numpy(np.array(inputs))
			am_labels = torch.from_numpy(np.array(am_labels))
			ml_labels = torch.from_numpy(np.array(ml_labels))

			if args.gpu:
				inputs = inputs.cuda()
				am_labels = am_labels.cuda()
				ml_labels = ml_labels.cuda()
				model = model.cuda()

			output = model(inputs, attention_mask=am_labels, masked_lm_labels=ml_labels)
			loss = output[0]
			if args.gpu:
				loss = loss.cuda()
			# training
			optim.zero_grad()
			loss.backward()
			clip_grad_norm(model.parameters(), args.max_grad_norm)
			tr_loss += loss.item()
			scheduler.step()
			optim.step()
			global_step += 1

			if global_step % 10000 == 0 :
				output_dir = os.path.join(args.save_path, 'checkpoint-{}'.format(global_step))
				if not os.path.exists(output_dir):
					os.makedirs(output_dir)
				model_to_save = model.module if hasattr(model, 'module') else model
				model_to_save.save_pretrained(output_dir)
				torch.save(args, os.path.join(output_dir, 'training_args.bin'))
			if exp_no%5000 == 0:
				print("{} batches trained.".format(exp_no))
		end_time = time.time()
		print("Epoch {} runs {} time.".format(epoch_no, (end_time-start_time)))
	f.close()
	return global_step, tr_loss/global_step

def main():

	model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
	tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
	#tokenizer.add_tokens([NUM_MASK, ENT_MASK])
	#NUM_ID = tokenizer.encode(NUM_MASK, add_special_tokens=False)[0]
	#ENT_ID = tokenizer.encode(ENT_MASK, add_special_tokens=False)[0]
	config = DistilBertConfig.from_pretrained("distilbert-base-uncased")


	if args.mode == "train":
		global_step, tr_loss = train(args, model, tokenizer)
		print(" global_step = {}, average loss = {}".format(global_step, tr_loss))

	if args.mode == 'train':
		if not os.path.exists(args.save_path):
			os.makedirs(args.save_path)
		print("Saving model checkpoint to {}".format(args.save_path))
		model_to_save = model.module if hasattr(model, 'module') else model
		model_to_save.save_pretrained(args.save_path)
		tokenizer.save_pretrained(args.save_path)

		torch.save(args, os.path.join(args.save_path, 'training_args.bin'))


if __name__ == "__main__":
	main()
