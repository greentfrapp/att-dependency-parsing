"""
Preprocessing and utility functions
"""


import numpy as np
import json
from collections import Counter


class DependencyTask(object):

	def __init__(self, filepath=None, dictpath=None, remove_punct=True):
		super(DependencyTask, self).__init__()
		self.idx = 0
		if filepath is not None:
			self.line_no = []
			self.x_tokens = []
			self.x_pos = []
			self.y_heads = []
			self.y_labels = []
			with open(filepath, 'r', encoding='utf-8') as file:
				idx = -1
				tokens = []
				tokens_pos = []
				heads = []
				labels = []
				for i, line in enumerate(file):
					if (i + 1) % 10000 == 0:
						print("Processing line {}...".format(i + 1))
					if line.startswith("#"):
						if len(tokens) > 0:
							self.line_no.append(idx)
							self.x_tokens.append(tokens)
							self.x_pos.append(tokens_pos)
							self.y_heads.append(heads)
							self.y_labels.append(labels)
							idx = -1
							tokens = []
							tokens_pos = []
							heads = []
							labels = []
					elif line.startswith('\n'):
						pass
					else:
						if idx == -1:
							idx = i
						elements = line.split('\t')
						token = elements[1]
						pos = elements[3]
						head = elements[6]
						label = elements[7]

						if head != '_':
							if remove_punct:
								if pos != 'PUNCT' and pos != 'SYM':
									tokens.append(token)
									tokens_pos.append(pos)
									heads.append(int(head))
									labels.append(label)
							else:
								tokens.append(token)
								tokens_pos.append(pos)
								heads.append(int(head))
								labels.append(label)

				if len(tokens) > 0:
					self.line_no.append(idx)
					self.x_tokens.append(tokens)
					self.x_pos.append(tokens_pos)
					self.y_heads.append(heads)
					self.y_labels.append(labels)
					idx = -1
					tokens = []
					tokens_pos = []
					heads = []
					labels = []

			if dictpath is None:
				dictpath = filepath + '.dict'
			self.make_dict(dictpath)

			with open("data/pos_dict.json", 'r') as file:
				self.pos_dict = json.load(file)

	def next_batch(self, batchsize=64, max_len=20):
		start = self.idx
		end = self.idx + batchsize
		minibatch_x_tokens = self.x_tokens[start:end]
		minibatch_x_pos = self.x_pos[start:end]
		minibatch_y_heads = self.y_heads[start:end]
		minibatch_y_labels = self.y_labels[start:end]
		self.idx = end
		if self.idx > len(self.x_tokens):
			self.idx = 0
			shuffler = np.random.RandomState(0)
			shuffler.shuffle(self.x_tokens)
			shuffler = np.random.RandomState(0)
			shuffler.shuffle(self.x_pos)
			shuffler = np.random.RandomState(0)
			shuffler.shuffle(self.y_heads)
			shuffler = np.random.RandomState(0)
			shuffler.shuffle(self.y_labels)
		return self.preprocess(minibatch_x_tokens, minibatch_x_pos, minibatch_y_heads, minibatch_y_labels, max_len)

	def preprocess(self, x_tokens, x_pos, y_heads, y_labels, max_len=20):
		new_x_tokens = []
		new_x_pos = []
		new_y_heads = []
		new_y_labels = []
		for i, _ in enumerate(x_tokens):
			tokens = []
			padded_sentence = ['<ROOT>']
			padded_sentence += x_tokens[i][:max_len]
			while len(padded_sentence) <= max_len:
				padded_sentence += ['<PAD>']
			padded_sentence += ['<NULL>']
			for token in padded_sentence:
				try:
					tokens.append(self.dict.index(token))
				except:
					tokens.append(self.dict.index('<UNK>'))
			new_x_tokens.append(np.eye(len(self.dict))[tokens])

			tokens_pos = []
			padded_pos = ['<ROOT>']
			padded_pos += x_pos[i][:max_len]
			while len(padded_pos) <= max_len:
				padded_pos += ['<PAD>']
			padded_pos += ['<NULL>']
			for pos in padded_pos:
				tokens_pos.append(self.pos_dict.index(pos))
			new_x_pos.append(np.eye(len(self.pos_dict))[tokens_pos])

			padded_heads = [-1]
			for head in y_heads[i][:max_len]:
				if head > max_len - 1:
					padded_heads.append(-1)
				else:
					padded_heads.append(head)
			while len(padded_heads) <= max_len:
				padded_heads.append(-1)
			padded_heads.append(-1)
			new_y_heads.append(np.eye(max_len + 2)[np.array(padded_heads)])
		return np.array(new_x_tokens), np.array(new_x_pos), np.array(new_y_heads)

	def make_dict(self, filepath, min_freq=10):
		try:
			with open(filepath, 'r') as file:
				self.dict = json.load(file)
		except:
			all_tokens = []
			for sentence in self.x_tokens:
				for token in sentence:
					all_tokens.append(token)
			all_tokens = Counter(all_tokens)
			self.dict = ['<PAD>', '<UNK>', '<ROOT>','<NULL>']
			for token, count in all_tokens.items():
				if count >= min_freq:
					self.dict.append(token)
			with open(filepath + '.dict', 'w') as file:
				json.dump(self.dict, file)
		# self.dict = np.array(self.dict)

	def export_json(self, filepath):
		with open(filepath + "_x_tokens", 'w') as file:
			json.dump(list(self.x_tokens), file)
		with open(filepath + "_y_heads", 'w') as file:
			json.dump(list(self.y_heads), file)
		with open(filepath + "_y_labels", 'w') as file:
			json.dump(list(self.y_labels), file)
		print("Files saved.")

	def load_json(self, filepath):
		with open(filepath + "_x_tokens", 'r') as file:
			self.x_tokens = json.load(file)
		with open(filepath + "_y_heads", 'r') as file:
			self.y_heads = json.load(file)
		with open(filepath + "_y_labels", 'r') as file:
			self.y_labels = json.load(file)
		print("Files loaded.")
