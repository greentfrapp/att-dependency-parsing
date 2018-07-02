
import numpy as np
import tensorflow as tf
from absl import flags
from absl import app

from utils import DependencyTask
from models import AttentionModel

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_integer("sample", 0, "Sample to test")

# Train parameters
flags.DEFINE_string("savepath", "models/", "Savepath to load/save models")

# Task parameters
flags.DEFINE_integer("max_len", 20, "Max sequence length")

def main(unused_args):
	if FLAGS.train:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=FLAGS.max_len)

		sess.run(tf.global_variables_initializer())

		n_steps = 5000

		for step in np.arange(n_steps):
			x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = task.next_batch(64, FLAGS.max_len)

			feed_dict = {
				model.inputs: x_tokens,
				model.pos_inputs: x_pos,
				model.pos_2_inputs: x_pos_2,
				model.y_in_parents: y_in_parents,
				model.y_in_children: y_in_children,
				model.y_out_parents: y_out_parents,
				model.y_out_children: y_out_children,
				model.is_training: True,
				# model.mask: mask,
			}
			
			loss, _ = sess.run([model.loss, model.optimize], feed_dict)
			if (step + 1) % 10 == 0:
				model.save(FLAGS.savepath, global_step=step+1)
				x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = dev_task.next_batch(64, FLAGS.max_len)

				feed_dict = {
					model.inputs: x_tokens,
					model.pos_inputs: x_pos,
					model.pos_2_inputs: x_pos_2,
					model.y_in_parents: y_in_parents,
					model.y_in_children: y_in_children,
					model.y_out_parents: y_out_parents,
					model.y_out_children: y_out_children,
					model.is_training: True,
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
				print("Step #{} - Loss : {:.2f} - Val Loss : {:.2f} - Val Acc : {}".format(step + 1, loss, val_loss, None))
				
				

	if FLAGS.test:
		task = DependencyTask("data/en_ewt-ud-train.conllu")
		dev_task = DependencyTask("data/en_ewt-ud-dev.conllu", dictpath="data/en_ewt-ud-train.conllu.dict")
		sess = tf.Session()
		model = AttentionModel(sess, len(task.dict), max_len=FLAGS.max_len)

		model.load(FLAGS.savepath)

		x_tokens, x_pos, x_pos_2, y_in_parents, y_in_children, y_out_parents, y_out_children = task.next_batch(64, FLAGS.max_len)

		correct = 0.
		wrong = 0.

		for n_sample in np.arange(64):

			gold_parents = dict()
			for i, token in enumerate(y_out_children[n_sample]):
				if np.argmax(token) != FLAGS.max_len + 1:
					gold_parents[np.argmax(token)] = np.argmax(y_out_parents[n_sample][i])

			token_ids = list(np.argmax(x_tokens[n_sample], axis=1))
			if 0 in token_ids:
				size = token_ids.index(0) 
			else:
				size = FLAGS.max_len + 1

			mask_children = np.ones(FLAGS.max_len + 2)
			mask_parents = np.ones(FLAGS.max_len + 2)
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
				while len(in_parents) < FLAGS.max_len:
					in_parents.append(-1)
					in_children.append(-1)

				if len(in_parents) > FLAGS.max_len:
					break

				feed_dict = {
					model.inputs: [x_tokens[n_sample]],
					model.pos_inputs: [x_pos[n_sample]],
					model.pos_2_inputs: [x_pos_2[n_sample]],
					model.y_in_parents: [np.eye(FLAGS.max_len + 2)[np.array(in_parents)]],
					model.y_in_children: [np.eye(FLAGS.max_len + 2)[np.array(in_children)]],
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

				mask_parents = np.zeros(FLAGS.max_len + 2)
				recursive_parent = new_child
				while recursive_parent != 0:
					mask_parents[recursive_parent] = 1
					recursive_parent = parents[children.index(recursive_parent)]
				mask_parents[0] = 1

				mask_children = np.zeros(FLAGS.max_len + 2)
				for child in children_left:
					mask_children[child] = 1
			
			predicted_parents = dict()
			for i, token in enumerate(children[1:]):
				predicted_parents[token] = parents[1:][i]

			for child, gold_parent in gold_parents.items():
				try:
					if gold_parent == predicted_parents[child]:
						correct += 1
					else:
						wrong += 1
				except:
					wrong += 1

			# sentence = dev_task.prettify(x_tokens[n_sample], clip_padding=True)
			# print(sentence)
			# print(parents[1:])
			# print(list(np.argmax(y_out_parents[n_sample], axis=1)))
			# print(children[1:])
			# print(list(np.argmax(y_out_children[n_sample], axis=1)))
			# quit()
		print(correct / (correct + wrong))


if __name__ == "__main__":
	app.run(main)

# Try predicting inside-out sequence but without repetition
# Use a second network to predict heads based on predicted sequence