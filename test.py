import numpy as np

from utils import DependencyTask

task = DependencyTask("data/en_ewt-ud-train.conllu")
total = 0
for bucket in task.buckets:
	print(len(task.buckets[bucket]))
	total += len(task.buckets[bucket])
print(total)
print(len(task.x_tokens))

x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = task.next_batch(64, 0)

# idx = np.random.choice(64)
idx = 0
print(np.argmax(x_tokens[idx], axis=1))
for token_id in np.argmax(x_tokens[idx], axis=1):
	print(task.dict[token_id])

print(np.argmax(y_out_parents[idx], axis=1))
print(np.argmax(y_out_children[idx], axis=1))