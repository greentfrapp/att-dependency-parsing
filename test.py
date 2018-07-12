import numpy as np

from utils import DependencyTask

task = DependencyTask("data/en_ewt-ud-dev.conllu")
# total = 0
# for bucket in task.buckets:
# 	print(len(task.buckets[bucket]))
# 	total += len(task.buckets[bucket])
# print(total)
# print(len(task.x_tokens))

for i in range(6):
	x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = task.next_batch(64, 2)

	print(np.array(x_tokens).shape)
	print(np.array(y_in_parents).shape)
	if i == 5:
		print(x_tokens)