"""
Preprocessing and utility functions
"""


import numpy as np
import json


class DependencyTask(object):

	def __init__(self, filepath=None, remove_punct=True):
		super(DependencyTask, self).__init__()
		if filepath is not None:
			self.line_no = []
			self.x_tokens = []
			self.y_heads = []
			self.y_labels = []
			with open(filepath, 'r') as file:
				idx = -1
				tokens = []
				heads = []
				labels = []
				for i, line in enumerate(file):
					if (i + 1) % 10000 == 0:
						print("Processing line {}...".format(i + 1))
					if line.startswith("#"):
						if len(tokens) > 0:
							self.line_no.append(idx)
							self.x_tokens.append(tokens)
							self.y_heads.append(heads)
							self.y_labels.append(labels)
							idx = -1
							tokens = []
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
									heads.append(int(head))
									labels.append(label)
							else:
								tokens.append(token)
								heads.append(int(head))
								labels.append(label)

				if len(tokens) > 0:
					self.line_no.append(idx)
					self.x_tokens.append(tokens)
					self.y_heads.append(heads)
					self.y_labels.append(labels)
					idx = -1
					tokens = []
					heads = []
					labels = []

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
