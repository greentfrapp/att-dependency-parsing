import numpy as np

from utils import DependencyTask

task = DependencyTask("data/en_ewt-ud-train.conllu")
# total = 0
# for bucket in task.buckets:
# 	print(len(task.buckets[bucket]))
# 	total += len(task.buckets[bucket])
# print(total)
# print(len(task.x_tokens))

for i in range(1):
	words, x_tokens, x_pos, x_pos_2, y_in, y_out = task.next_batch(64, 1)

	for i, sample in enumerate(y_out[:5]):
		print(words[i])
		print(np.argmax(sample, axis=1))