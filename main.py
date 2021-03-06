
import numpy as np
import tensorflow as tf
from absl import flags
from absl import app

from utils import DependencyTask
from models import AttentionModel, AttentionModel2

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

flags.DEFINE_bool("load", False, "Load to resume training")
flags.DEFINE_integer("sample", 0, "Sample to test")

# Train parameters
flags.DEFINE_string("savepath", "models/", "Savepath to load/save models")
flags.DEFINE_integer("steps", 500, "Number of steps to train model")

# Task parameters
# flags.DEFINE_integer("max_len", 40, "Max sequence length")
flags.DEFINE_integer("bucket_id", 1, "Bucket id to select for training")

def main(unused_args):
	if FLAGS.train:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=task.bucket_lengths[FLAGS.bucket_id])
		# model = AttentionModel(sess, len(task.dict), max_len=140)

		sess.run(tf.global_variables_initializer())
		if FLAGS.load:
			model.load(FLAGS.savepath, verbose=True)
		# else:
		
		min_val_loss = np.inf

		for step in np.arange(FLAGS.steps):
			x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = task.next_batch(64, FLAGS.bucket_id)
			# x_tokens, x_pos, x_pos_2, y_in, y_out = task.next_batch(64, FLAGS.bucket_id)

			feed_dict = {
				model.inputs: x_tokens,
				model.pos_inputs: x_pos,
				# model.pos_2_inputs: x_pos_2,
				model.y_in_parents: y_in_parents,
				model.y_in_children: y_in_children,
				model.y_out_parents: y_out_parents,
				model.y_out_children: y_out_children,
				# model.y_in: y_in,
				# model.y_out: y_out,
				model.is_training: True,
				# model.mask: mask,
			}
			
			# att, orig = sess.run([model.attention, model.softmax], feed_dict)
			# print(att.shape)
			# print(orig.shape)
			# quit()

			loss, _ = sess.run([model.loss, model.optimize], feed_dict)
			if (step + 1) % 10 == 0:
				x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = dev_task.next_batch(64, FLAGS.bucket_id)
				# x_tokens, x_pos, x_pos_2, y_in, y_out = dev_task.next_batch(64, FLAGS.bucket_id)

				feed_dict = {
					model.inputs: x_tokens,
					model.pos_inputs: x_pos,
					# model.pos_2_inputs: x_pos_2,
					model.y_in_parents: y_in_parents,
					model.y_in_children: y_in_children,
					model.y_out_parents: y_out_parents,
					model.y_out_children: y_out_children,
					# model.y_in: y_in,
					# model.y_out: y_out,
					model.is_training: False,
					# model.mask: mask,
				}
				val_loss = sess.run(model.loss, feed_dict)
				# correct = 0.
				# wrong = 0.
				# for i, sample_pred in enumerate(val_pred):
				# 	for j, pred in enumerate(sample_pred):
				# 		if np.argmax(y_heads[i, j, :]) != FLAGS.max_len + 1:
				# 			if np.argmax(y_heads[i, j, :]) == pred:
				# 				correct += 1
				# 			else:
				# 				wrong += 1
				# for i, sample_pred in enumerate(val_pred):
				# 	if sample_pred[0] == np.argmax(y_heads[i, :]):
				# 		correct += 1
				# 	else:
				# 		wrong += 1
				print("Step #{} - Loss : {:.3f} - Val Loss : {:.3f} - Val Acc : {}".format(step + 1, loss, val_loss, None))

				if val_loss < min_val_loss:
					model.save(FLAGS.savepath, global_step=step+1)
					min_val_loss = val_loss
				
				

	if FLAGS.test:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		max_len = task.bucket_lengths[FLAGS.bucket_id]
		# max_len = 140
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=max_len)

		model.load(FLAGS.savepath)

		x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = dev_task.next_batch(64, FLAGS.bucket_id)
		# x_tokens, x_pos, x_pos_2, _, y_out = dev_task.next_batch(64, FLAGS.bucket_id)
	
		correct = 0.
		wrong = 0.

		for n_sample in np.arange(64):

			# gold_parents = {}
			# current_parent = 0
			# for token in y_out[n_sample]:
			# 	token_id = np.argmax(token)
			# 	if token_id == max_len + 1 or token_id == 0:
			# 		break
			# 	if token_id not in gold_parents:
			# 		gold_parents[token_id] = current_parent
			# 		current_parent = token_id
			# 	else:
			# 		current_parent = gold_parents[token_id]

			# seq = [0]
			# current_parent = 0
			# children_left = list(np.arange(1, max_len + 1))
			# parents = {}
			# while True:

			# 	mask = np.zeros(max_len + 2)
			# 	mask[current_parent] = 1
			# 	for child in children_left:
			# 		mask[child] = 1

			# 	y_in = []
			# 	for el in seq:
			# 		y_in.append(el)
			# 	while len(y_in) < (2 * max_len + 1):
			# 		y_in.append(max_len + 1)
				
			# 	feed_dict = {
			# 		model.inputs: [x_tokens[n_sample]],
			# 		model.pos_inputs: [x_pos[n_sample]],
			# 		model.y_in: [np.eye(max_len + 2)[np.array(y_in, dtype=np.int32)]],
			# 		model.is_training: False,
			# 	}

			# 	new = sess.run(model.softmax, feed_dict)[0, len(seq) - 1, :]
			# 	new = new * mask
			# 	new_id = np.argmax(new)

			# 	seq.append(new_id)

			# 	if new_id in children_left:
			# 		children_left.pop(children_left.index(new_id))
			# 		parents[new_id] = current_parent
			# 		current_parent = new_id
			# 	elif new_id == 0:
			# 		break
			# 	else:
			# 		current_parent = parents[new_id]

			# print()
			# print(parents)
			# print(gold_parents)


			gold_parents = dict()
			for i, token in enumerate(y_out_children[n_sample]):
				if np.argmax(token) != max_len + 1:
					gold_parents[np.argmax(token)] = np.argmax(y_out_parents[n_sample][i])

			token_ids = list(np.argmax(x_tokens[n_sample], axis=1))
			if 0 in token_ids:
				size = token_ids.index(0)
			else:
				size = max_len + 1

			mask_children = np.ones(max_len + 2)
			mask_parents = np.zeros(max_len + 2)
			mask_children[-1] = 0
			mask_parents[0] = 1
			children_left = list(np.arange(1, size))
			parents = [0]
			children = [0]
			n = 0
			while True:
				in_parents = []
				in_children = []
				for i, _ in enumerate(parents):
					in_parents.append(parents[i])
					in_children.append(children[i])
				while len(in_parents) < max_len:
					in_parents.append(-1)
					in_children.append(-1)

				if len(in_parents) > max_len:
					break

				feed_dict = {
					model.inputs: [x_tokens[n_sample]],
					model.pos_inputs: [x_pos[n_sample]],
					model.pos_2_inputs: [x_pos_2[n_sample]],
					model.y_in_parents: [np.eye(max_len + 2)[np.array(in_parents)]],
					model.y_in_children: [np.eye(max_len + 2)[np.array(in_children)]],
					model.is_training: False,
					# model.mask: mask,
				}
				softmax_parents, softmax_children = sess.run([model.softmax_parents, model.softmax_children], feed_dict)
				
				new_parent = np.argmax(softmax_parents[0][n] * mask_parents)
				new_child = np.argmax(softmax_children[0][n] * mask_children)
				parents.append(new_parent)
				children.append(new_child)
				n += 1
				if new_child in children_left:
					children_left.pop(children_left.index(new_child))

				if len(children_left) == 0:
					break

				mask_parents = np.zeros(max_len + 2)
				recursive_parent = new_child
				while recursive_parent != 0:
					mask_parents[recursive_parent] = 1
					recursive_parent = parents[children.index(recursive_parent)]
				mask_parents[0] = 1

				mask_children = np.zeros(max_len + 2)
				for child in children_left:
					mask_children[child] = 1
			
			predicted_parents = dict()
			for i, token in enumerate(children[1:]):
				predicted_parents[token] = parents[1:][i]

			for child, gold_parent in gold_parents.items():
				try:
					if gold_parent == parents[child]:
						correct += 1
					else:
						wrong += 1
				except:
					wrong += 1

		# 	print("Predicting sample #{}...".format(n_sample))
		# 	print(predicted_parents)
		# 	print(gold_parents)
			
		# 	# sentence = dev_task.prettify(x_tokens[n_sample], clip_padding=True)
		# 	# print(sentence)
		# 	# print(parents[1:])
		# 	# print(list(np.argmax(y_out_parents[n_sample], axis=1)))
		# 	# print(children[1:])
		# 	# print(list(np.argmax(y_out_children[n_sample], axis=1)))
		# 	# quit()
		print(correct / (correct + wrong))


if __name__ == "__main__":
	app.run(main)

# Try predicting inside-out sequence but without repetition
# Use a second network to predict heads based on predicted sequence