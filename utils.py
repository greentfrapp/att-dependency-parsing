"""
Preprocessing and utility functions
"""

import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


class DependencyTask(object):

	def __init__(self, filepath=None, dictpath=None, remove_punct=True):
		super(DependencyTask, self).__init__()
		self.idx = 0
		# self.bucket_lengths = [1, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]
		self.bucket_lengths = [1, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]
		self.buckets = {}
		for length in self.bucket_lengths:
			self.buckets[length] = []
		if filepath is not None:
			self.line_no = []
			self.x_tokens = []
			self.x_pos = []
			self.x_pos_2 = []
			self.y_heads = []
			self.y_labels = []
			with open(filepath, 'r', encoding='utf-8') as file:
				idx = 0
				punct = []
				tokens = []
				tokens_pos = []
				tokens_pos_2 = []
				heads = []
				labels = []
				for i, line in enumerate(file):
					if (i + 1) % 10000 == 0:
						print("Processing line {}...".format(i + 1))
					if line.startswith("#"):
						if len(tokens) > 0:
							mod = np.zeros_like(heads)
							for n in punct:
								for i, head in enumerate(heads):
									if head > n:
										mod[i] += 1
							heads = list(np.array(heads) - mod)
							self.line_no.append(idx)
							self.x_tokens.append(tokens)
							self.x_pos.append(tokens_pos)
							self.x_pos_2.append(tokens_pos_2)
							self.y_heads.append(heads)
							self.y_labels.append(labels)
						idx = 0
						punct = []
						tokens = []
						tokens_pos = []
						tokens_pos_2 = []
						heads = []
						labels = []
					elif line.startswith('\n'):
						pass
					else:
						idx += 1
						elements = line.split('\t')
						token = elements[1]
						pos = elements[3]
						pos_2 = elements[4]
						head = elements[6]
						label = elements[7]

						if head != '_':
							if remove_punct:
								if pos != 'PUNCT' and pos != 'SYM':
									tokens.append(token)
									tokens_pos.append(pos)
									tokens_pos_2.append(pos_2)
									heads.append(int(head))
									labels.append(label)
								else:
									punct.append(idx)
							else:
								tokens.append(token)
								tokens_pos.append(pos)
								tokens_pos_2.append(pos_2)
								heads.append(int(head))
								labels.append(label)

				if len(tokens) > 0:
					mod = np.zeros_like(heads)
					for n in punct:
						for i, head in enumerate(heads):
							if head > n:
								mod[i] += 1
					heads = list(np.array(heads) - mod)
					self.line_no.append(idx)
					self.x_tokens.append(tokens)
					self.x_pos.append(tokens_pos)
					self.x_pos_2.append(tokens_pos_2)
					self.y_heads.append(heads)
					self.y_labels.append(labels)

			self.x_tokens = np.array(self.x_tokens)
			self.x_pos = np.array(self.x_pos)
			self.x_pos_2 = np.array(self.x_pos_2)
			self.y_heads = np.array(self.y_heads)
			self.y_labels = np.array(self.y_labels)

			# Sort samples into buckets
			for i, sequence in enumerate(self.x_tokens):
				for length in self.bucket_lengths:
					if len(sequence) <= length:
						self.buckets[length].append(i)
						break

			if dictpath is None:
				dictpath = filepath + '.dict'
			self.make_dict(dictpath)

			with open("data/pos_dict.json", 'r') as file:
				self.pos_dict = json.load(file)
			with open("data/pos_2_dict.json", 'r') as file:
				self.pos_2_dict = json.load(file)

			self.wv = KeyedVectors.load("data/en_ewt-ud-train.conllu.kv", mmap='r')

	def next_batch(self, batchsize=64, bucket=0):
		start = self.idx
		end = self.idx + batchsize
		minibatch_bucket = np.array(self.buckets[self.bucket_lengths[bucket]][start:end], dtype=np.int32)
		minibatch_x_tokens = self.x_tokens[minibatch_bucket]
		minibatch_x_pos = self.x_pos[minibatch_bucket]
		minibatch_x_pos_2 = self.x_pos_2[minibatch_bucket]
		minibatch_y_heads = self.y_heads[minibatch_bucket]
		minibatch_y_labels = self.y_labels[minibatch_bucket]
		self.idx = end
		if self.idx >= len(self.buckets[self.bucket_lengths[bucket]]):
			self.idx = 0
			shuffler = np.random.RandomState(0)
			shuffler.shuffle(self.buckets[self.bucket_lengths[bucket]])
		return self.preprocess4(minibatch_x_tokens, minibatch_x_pos, minibatch_x_pos_2, minibatch_y_heads, minibatch_y_labels, max_len=self.bucket_lengths[bucket])
		# return self.preprocess(minibatch_x_tokens, minibatch_x_pos, minibatch_x_pos_2, minibatch_y_heads, minibatch_y_labels, max_len=140)

	def preprocess(self, x_tokens, x_pos, x_pos_2, y_heads, y_labels, max_len):

		new_x_tokens = []
		new_x_pos = []
		new_x_pos_2 = []
		new_y_in_parents = []
		new_y_in_children = []
		new_y_out_parents = []
		new_y_out_children = []

		for i, _ in enumerate(x_tokens):

			# Onehot encode tokens
			# Add <ROOT> at beginning, <PAD> to max_len and add <NULL> at end
			tokens = [self.dict.index('<ROOT>')]
			for token in x_tokens[i][:max_len]:
				try:
					tokens.append(self.dict.index(token))
				except:
					tokens.append(self.dict.index('<UNK>'))
			while len(tokens) < max_len + 1:
				tokens.append(self.dict.index('<PAD>'))
			tokens.append(self.dict.index('<NULL>'))

			# Onehot encode POS tags
			pos_onehot = [self.pos_dict.index('<ROOT>')]
			for pos in x_pos[i][:max_len]:
				pos_onehot.append(self.pos_dict.index(pos))
			while len(pos_onehot) < max_len + 1:
				pos_onehot.append(self.pos_dict.index('<PAD>'))
			pos_onehot.append(self.pos_dict.index('<NULL>'))

			# Onehot encode POS 2 tags
			pos_2_onehot = [self.pos_2_dict.index('<ROOT>')]
			for pos in x_pos_2[i][:max_len]:
				pos_2_onehot.append(self.pos_2_dict.index(pos))
			while len(pos_2_onehot) < max_len + 1:
				pos_2_onehot.append(self.pos_2_dict.index('<PAD>'))
			pos_2_onehot.append(self.pos_2_dict.index('<NULL>'))

			# Flattened parse tree, depth first
			parse_tree_children = [0]
			parse_tree_parents = [0]
			children = list(np.arange(1, max_len + 1))
			current_parent = 0
			while len(children) > 0:
				# current_parent = parse_tree[-1]
				found_child = False
				for k, head in enumerate(y_heads[i][:max_len]):
					if head == current_parent and (k + 1) in children:
						parse_tree_children.append(k + 1)
						parse_tree_parents.append(current_parent)
						children.pop(children.index(k + 1))
						found_child = True
						current_parent = k + 1
						break
				if not found_child:
					if current_parent != 0:
						# parse_tree.append(list(y_heads[i][:max_len])[current_parent - 1])
						current_parent = list(y_heads[i][:max_len])[current_parent - 1]
					else:
						break
			
			while len(parse_tree_parents) < max_len + 1:
				parse_tree_parents.append(max_len + 1)
			while len(parse_tree_children) < max_len + 1:
				parse_tree_children.append(max_len + 1)

			new_x_tokens.append(np.eye(len(self.dict))[np.array(tokens)])
			new_x_pos.append(np.eye(len(self.pos_dict))[np.array(pos_onehot)])
			new_x_pos_2.append(np.eye(len(self.pos_2_dict))[np.array(pos_2_onehot)])
			new_y_in_parents.append(np.eye(max_len + 2)[np.array(parse_tree_parents[:-1])])
			new_y_in_children.append(np.eye(max_len + 2)[np.array(parse_tree_children[:-1])])
			new_y_out_parents.append(np.eye(max_len + 2)[np.array(parse_tree_parents[1:])])
			new_y_out_children.append(np.eye(max_len + 2)[np.array(parse_tree_children[1:])])

		return new_x_tokens, new_x_pos, new_x_pos_2, new_y_in_parents, new_y_in_children, new_y_out_parents, new_y_out_children

	def preprocess2(self, x_tokens, x_pos, x_pos_2, y_heads, y_labels, max_len):

		new_x_tokens = []
		new_x_pos = []
		new_x_pos_2 = []
		new_y_in = []
		new_y_out = []

		for i, _ in enumerate(x_tokens):

			# Onehot encode tokens
			# Add <ROOT> at beginning, <PAD> to max_len and add <NULL> at end
			tokens = [self.dict.index('<ROOT>')]
			for token in x_tokens[i][:max_len]:
				try:
					tokens.append(self.dict.index(token))
				except:
					tokens.append(self.dict.index('<UNK>'))
			while len(tokens) < max_len + 1:
				tokens.append(self.dict.index('<PAD>'))
			tokens.append(self.dict.index('<NULL>'))

			# Onehot encode POS tags
			pos_onehot = [self.pos_dict.index('<ROOT>')]
			for pos in x_pos[i][:max_len]:
				pos_onehot.append(self.pos_dict.index(pos))
			while len(pos_onehot) < max_len + 1:
				pos_onehot.append(self.pos_dict.index('<PAD>'))
			pos_onehot.append(self.pos_dict.index('<NULL>'))

			# Onehot encode POS 2 tags
			pos_2_onehot = [self.pos_2_dict.index('<ROOT>')]
			for pos in x_pos_2[i][:max_len]:
				pos_2_onehot.append(self.pos_2_dict.index(pos))
			while len(pos_2_onehot) < max_len + 1:
				pos_2_onehot.append(self.pos_2_dict.index('<PAD>'))
			pos_2_onehot.append(self.pos_2_dict.index('<NULL>'))

			# Flattened parse tree, inside-out, left-right
			parse_tree = [0]
			children = list(np.arange(1, len(x_tokens[i])))
			current_parent = 0
			while len(children) > 0:
				# current_parent = parse_tree[-1]
				left = []
				right = []
				for child in children:
					if child < current_parent:
						left.append(child)
					else:
						right.append(child)
				children = left[::-1] + right
				found_child = False
				for child in children:
					if y_heads[i][child - 1] == current_parent:
						parse_tree.append(child)
						children.pop(children.index(child))
						found_child = True
						current_parent = child
						break

				if not found_child:
					if current_parent != 0:
						parse_tree.append(current_parent)
						current_parent = y_heads[i][current_parent - 1]
					else:
						break

				if current_parent == 0:
					parse_tree.append(0)
					break
			
			while len(parse_tree) < 2 * max_len + 1:
				parse_tree.append(max_len + 1)

			new_x_tokens.append(np.eye(len(self.dict))[np.array(tokens)])
			new_x_pos.append(np.eye(len(self.pos_dict))[np.array(pos_onehot)])
			new_x_pos_2.append(np.eye(len(self.pos_2_dict))[np.array(pos_2_onehot)])
			new_y_in.append(np.eye(max_len + 2)[np.array(parse_tree[:-1])])
			new_y_out.append(np.eye(max_len + 2)[np.array(parse_tree[1:])])

		return new_x_tokens, new_x_pos, new_x_pos_2, new_y_in, new_y_out

	def preprocess3(self, x_tokens, x_pos, x_pos_2, y_heads, y_labels, max_len):

		new_x_tokens = []
		new_x_pos = []
		new_x_pos_2 = []
		new_y_in = []
		new_y_out = []

		for i, _ in enumerate(x_tokens):

			# Onehot encode tokens
			# Add <ROOT> at beginning, <PAD> to max_len and add <NULL> at end
			tokens = [self.dict.index('<ROOT>')]
			for token in x_tokens[i]:
				try:
					tokens.append(self.dict.index(token))
				except:
					tokens.append(self.dict.index('<UNK>'))
			while len(tokens) < max_len + 1:
				tokens.append(self.dict.index('<PAD>'))
			tokens.append(self.dict.index('<NULL>'))

			tokens_wv = []
			for token in tokens:
				if token < 4:
					tokens_wv.append(np.eye(100)[token])
				else:
					tokens_wv.append(self.wv[self.dict[token]])

			# Onehot encode POS tags
			pos_onehot = [self.pos_dict.index('<ROOT>')]
			for pos in x_pos[i][:max_len]:
				pos_onehot.append(self.pos_dict.index(pos))
			while len(pos_onehot) < max_len + 1:
				pos_onehot.append(self.pos_dict.index('<PAD>'))
			pos_onehot.append(self.pos_dict.index('<NULL>'))

			# Onehot encode POS 2 tags
			pos_2_onehot = [self.pos_2_dict.index('<ROOT>')]
			for pos in x_pos_2[i][:max_len]:
				pos_2_onehot.append(self.pos_2_dict.index(pos))
			while len(pos_2_onehot) < max_len + 1:
				pos_2_onehot.append(self.pos_2_dict.index('<PAD>'))
			pos_2_onehot.append(self.pos_2_dict.index('<NULL>'))

			# Flattened parse tree, inside-out, left-right
			parse_tree = [0]
			children = list(np.arange(1, len(x_tokens[i]) + 1))
			current_parent = 0
			while True:
				# current_parent = parse_tree[-1]
				left = []
				right = []
				for child in children:
					if child < current_parent:
						left.append(child)
					else:
						right.append(child)
				children = left[::-1] + right
				found_child = False
				for child in children:
					if y_heads[i][child - 1] == current_parent:
						parse_tree.append(child)
						children.pop(children.index(child))
						found_child = True
						current_parent = child
						break

				if not found_child:
					if current_parent != 0:
						parse_tree.append(current_parent)
						current_parent = y_heads[i][current_parent - 1]
					else:
						break

				if current_parent == 0:
					parse_tree.append(0)
					break
			
			while len(parse_tree) < 2 * max_len + 2:
				parse_tree.append(max_len + 1)

			# new_x_tokens.append(np.eye(len(self.dict))[np.array(tokens)])
			new_x_tokens.append(tokens_wv)
			new_x_pos.append(np.eye(len(self.pos_dict))[np.array(pos_onehot)])
			new_x_pos_2.append(np.eye(len(self.pos_2_dict))[np.array(pos_2_onehot)])
			new_y_in.append(np.eye(max_len + 2)[np.array(parse_tree[:-1])])
			new_y_out.append(np.eye(max_len + 2)[np.array(parse_tree[1:])])

		return new_x_tokens, new_x_pos, new_x_pos_2, new_y_in, new_y_out

	def preprocess4(self, x_tokens, x_pos, x_pos_2, y_heads, y_labels, max_len):

		new_x_tokens = []
		new_x_pos = []
		new_x_pos_2 = []
		new_y_in_parents = []
		new_y_in_children = []
		new_y_out_parents = []
		new_y_out_children = []

		for i, _ in enumerate(x_tokens):

			# Onehot encode tokens
			# Add <ROOT> at beginning, <PAD> to max_len and add <NULL> at end
			tokens = [self.dict.index('<ROOT>')]
			for token in x_tokens[i][:max_len]:
				try:
					tokens.append(self.dict.index(token))
				except:
					tokens.append(self.dict.index('<UNK>'))
			while len(tokens) < max_len + 1:
				tokens.append(self.dict.index('<PAD>'))
			tokens.append(self.dict.index('<NULL>'))

			tokens_wv = []
			for token in tokens:
				if token < 4:
					tokens_wv.append(np.eye(100)[token])
				else:
					tokens_wv.append(self.wv[self.dict[token]])

			# Onehot encode POS tags
			pos_onehot = [self.pos_dict.index('<ROOT>')]
			for pos in x_pos[i][:max_len]:
				pos_onehot.append(self.pos_dict.index(pos))
			while len(pos_onehot) < max_len + 1:
				pos_onehot.append(self.pos_dict.index('<PAD>'))
			pos_onehot.append(self.pos_dict.index('<NULL>'))

			# Onehot encode POS 2 tags
			pos_2_onehot = [self.pos_2_dict.index('<ROOT>')]
			for pos in x_pos_2[i][:max_len]:
				pos_2_onehot.append(self.pos_2_dict.index(pos))
			while len(pos_2_onehot) < max_len + 1:
				pos_2_onehot.append(self.pos_2_dict.index('<PAD>'))
			pos_2_onehot.append(self.pos_2_dict.index('<NULL>'))

			# Flattened parse tree, depth first
			parse_tree_children = [0]
			parse_tree_parents = [0]
			children = list(np.arange(1, max_len + 1))
			current_parent = 0
			while len(children) > 0:
				# current_parent = parse_tree[-1]
				found_child = False
				for k, head in enumerate(y_heads[i][:max_len]):
					if head == current_parent and (k + 1) in children:
						parse_tree_children.append(k + 1)
						parse_tree_parents.append(current_parent)
						children.pop(children.index(k + 1))
						found_child = True
						current_parent = k + 1
						break
				if not found_child:
					if current_parent != 0:
						# parse_tree.append(list(y_heads[i][:max_len])[current_parent - 1])
						current_parent = list(y_heads[i][:max_len])[current_parent - 1]
					else:
						break
			
			while len(parse_tree_parents) < max_len + 1:
				parse_tree_parents.append(max_len + 1)
			while len(parse_tree_children) < max_len + 1:
				parse_tree_children.append(max_len + 1)

			# new_x_tokens.append(np.eye(len(self.dict))[np.array(tokens)])
			new_x_tokens.append(tokens_wv)
			new_x_pos.append(np.eye(len(self.pos_dict))[np.array(pos_onehot)])
			new_x_pos_2.append(np.eye(len(self.pos_2_dict))[np.array(pos_2_onehot)])
			new_y_in_parents.append(np.eye(max_len + 2)[np.array(parse_tree_parents[:-1])])
			new_y_in_children.append(np.eye(max_len + 2)[np.array(parse_tree_children[:-1])])
			new_y_out_parents.append(np.eye(max_len + 2)[np.array(parse_tree_parents[1:])])
			new_y_out_children.append(np.eye(max_len + 2)[np.array(parse_tree_children[1:])])

		return new_x_tokens, new_x_pos, new_x_pos_2, new_y_in_parents, new_y_in_children, new_y_out_parents, new_y_out_children


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

	def prettify(self, tokens, clip_padding=True):
		tokens = np.array(self.dict)[np.argmax(tokens, axis=1)]
		while tokens[-1] == '<PAD>' or tokens[-1] == '<NULL>':
			tokens = tokens[:-1]
		return " ".join(tokens)

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

# if __name__ == "__main__":
# 	dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
# 	dev_task.next_batch(64, 20)
